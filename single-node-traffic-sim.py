"""Proof of concept of a single node car simulator.

This simulator connects all the components, including history together to test
the entire pipeline.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
from typing import List
from tqdm import tqdm
import click
import random
import redis
from probduration import HistoryHandler, Route, probable_duration

from ruth.data.border import Border, BorderType
from ruth.data.cz import County
from ruth.car import load_cars, Car


@dataclass
class CarRoutes:
    """Save car and assigned routes at one place."""

    car: Car
    osm_routes: List[List[int]]
    routes: List[Route]


def shortest_paths_segments(cars: List[Car], k: int) -> List[CarRoutes]:
    """Get the shortest paths for each car.

    It uses the `current_starting_node` and the `destination_node` for
    computation of the shortest path.
    """
    car_routes = []
    for car in cars:
        osm_routes = car.k_shortest_path(k)

        # TODO: 20s move among the parameters
        routes = list(
            map(lambda osm_route: Route(car.route_to_segments(osm_route),
                                        timedelta(seconds=20)),
                osm_routes))
        car_routes.append(CarRoutes(car, osm_routes, routes))

    return car_routes


def reorder(cars_k_routes: List[CarRoutes], departure_time, history, samples) -> (Car, Route):
    """Pick one route for for each group."""
    result = []
    for car_k_routes in cars_k_routes:

        if len(car_k_routes.routes) > 1:
            # compute reordering only if there is more then one route
            durations = map(lambda driving_route: probable_duration(
                driving_route, departure_time, history, samples), car_k_routes.routes)
            indexed_durations = sorted(enumerate(durations),
                                       key=lambda indexed_duration: indexed_duration[1])

            best_route_index, _ = indexed_durations[0]
        else:
            best_route_index = 0

        car = car_k_routes.car
        car.set_osm_route(car_k_routes.osm_routes[best_route_index])
        result.append((car, car_k_routes.routes[best_route_index]))
    return result


@click.command()
@click.argument("input_csv", type=click.Path(exists=True))
@click.option("--departure-time",
              type=click.DateTime(),
              default=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
@click.option("--k-routes",
              type=int,
              default=4,
              help="A number of alterantive routes routed between two points.")
@click.option("--n-samples",
              type=int,
              default=1000,
              help="A number of samples of monte carlo simulation.")
@click.option("--seed",
              type=int,
              help=("A fixed seed for random number generator"
                    "enusiring the same behaviour in the next run."))
@click.option("--redis-host", default="localhost")
@click.option("--redis-port", default=6379)
def simulate(input_csv, departure_time, k_routes, n_samples, seed, redis_host, redis_port):
    """Single-node traffic simulator."""
    redis_cli = redis.Redis(host=redis_host, port=redis_port)
    # flush the previous runs
    redis_cli.flushdb()

    if seed is not None:
        random.seed(seed)  # 660277

    # TODO: later use the real history
    history = HistoryHandler.no_limit()

    current_time_offset = timedelta(seconds=0)

    finished_cars = []
    cars_queue = load_cars(input_csv, _get_routing_border())

    with tqdm(total=float("inf")) as pbar:
        while cars_queue:
            # sort the cars according to the time offset
            cars_queue.sort(key=lambda car: car.departure_time_offset)
            time_groups = groupby(cars_queue, key=lambda car: car.departure_time_offset)

            # only the cars in the first group will be processed
            start_offset, cars = next(time_groups)
            cars = list(cars)

            # take the processed cars from the group out of the queue
            cars_queue = cars_queue[len(cars):]

            current_time_offset = start_offset
            pbar.set_description(
                (f"Processing {len(cars)} car{'' if len(cars) == 1 else 's'}"
                 f" at the offset +{current_time_offset}"))

            k_car_routes = shortest_paths_segments(cars, k_routes)

            dt = departure_time + current_time_offset
            cars_driving_route = reorder(k_car_routes, dt, history, n_samples)

            for car, driving_route in cars_driving_route:
                try:
                    time, segment_pos = driving_route.advance(
                        dt, car.segment_position, history, random.random())
                    d = time - dt
                    car.departure_time_offset = current_time_offset + d

                    car.advance(segment_pos)

                    osm_start_node_id = str(car.osm_route[segment_pos.index])
                    # store the current time offset in the milliseconds resolution
                    current_time_offset_ms = (current_time_offset + d) / timedelta(milliseconds=1)
                    redis_cli.zadd(
                        f"logical:{osm_start_node_id}",
                        {f"car_{car.id}": current_time_offset_ms})
                    # tracking history
                    redis_cli.zadd(
                        f"history:{osm_start_node_id}",
                        {f"car_{car.id}:{datetime.now().timestamp()}": current_time_offset_ms})

                    cars_queue.append(car)
                except StopIteration:
                    finished_cars.append((car.id, dt, car.osm_route))
            pbar.update(1)

        _process_finished_cars(finished_cars)


def _process_finished_cars(finished_cars):
    """Informative print of processed cars.

    TODO: Future version process the traffic stored in redis.
    """
    print(f"\n#cars {len(finished_cars)}")
    for car_id, arrival_time, route, in finished_cars:
        arrival_time_fmt = arrival_time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"The car {car_id} arrived at {arrival_time_fmt} and drove route: {route}.")


def _get_routing_border():
    """All the testing data are currently in Prague.

    For the future a kind of preprocessing must be done to pre-compute the routing areas
    in-advance (and once) as it's time-consuming process.
    """
    area = County.CZ010
    prague = Border(area.name, area.value, BorderType.COUNTY, "./data", True)
    return prague


if __name__ == "__main__":
    simulate()
