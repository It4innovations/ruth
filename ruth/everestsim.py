
import logging
import pandas as pd
import random
from ruth.vehicle import Vehicle
from probduration import HistoryHandler, Route, probable_duration
from ruth.utils import osm_route_to_segments


logger = logging.getLogger(__name__)


def load_vehicles(input_path: str):
    logger.info("Loading data ... %s", input_path)
    df = pd.read_pickle(input_path)
    return [Vehicle(**row.to_dict()) for (_, row) in df.iterrows()]


def simulate(input_path: str,
             departure_time,
             k_routes,
             n_samples,
             seed,
             gv_update_period,
             intermediate_results,
             checkpoint_period):

    all_vehicles = load_vehicles(input_path)
    active_vehicles = all_vehicles
    step = 0

    while active_vehicles:
        logger.info("Starting step %s", step)
        for i in range(gv_update_period): # compute the cars' leap
            logger.info("Update period %s/%s, #cars = %s", step, i, len(active_vehicles))
            min_offset = min(v.time_offset for v in active_vehicles)
            new_active_vehicles = []
            for vehicle in active_vehicles:
                if vehicle.time_offset == min_offset:
                    new_vehicle = advance_vehicle(vehicle, n_samples, k_routes, departure_time)
                    if new_vehicle.active:
                        new_active_vehicles.append(vehicle)
                else:
                    # I cannot skip the vehicles that time_offset is higher then min_offset and still active
                    new_active_vehicles.append(vehicle)
            active_vehicles = new_active_vehicles
            if not active_vehicles:
                break

        if intermediate_results is not None and step % checkpoint_period == 0:
            # TODO: Save vehicles
            pass
        step += 1


# EXACTLY SAME FUNCTION as in distsim, but decorator has been removed
def advance_vehicle(vehicle, samples, k_routes, departure_time):
    """Advance a vehicle on a route."""

    # compute the k shortest paths and compose driving rotes from them
    osm_routes = vehicle.k_shortest_paths(k_routes)  # TODO: unify using of _path_ and _route_ terms
    possible_driving_routes = list(
        map(lambda osm_route: Route(osm_route_to_segments(osm_route, vehicle.routing_map),
                                    vehicle.frequency),
            osm_routes))

    dt = departure_time + vehicle.time_offset
    history = HistoryHandler.no_limit()  # TODO: history get here or take as an argument?

    # pick the driving route with the smallest deylay
    if len(possible_driving_routes) > 1:
        delays = map(lambda driving_route: probable_duration(  # TODO: rename probable durations to probable delays
            driving_route, dt, history, samples), possible_driving_routes)
        indexed_delays = sorted(enumerate(delays), key=lambda indexed_delay: indexed_delay[1])

        best_route_index, _ = indexed_delays[0]
    else:
        best_route_index = 0

    # update the current route
    vehicle.set_current_route(osm_routes[best_route_index])

    # advance the vehicle on the driving route
    driving_route = possible_driving_routes[best_route_index]
    time, segment_pos = driving_route.advance(
        dt, vehicle.segment_position, history, random.random())
    d = time - dt

    # NOTE: _assumtion_: the car stays on a single segment within one call of the `advance`
    #       method on the driving route

    if segment_pos.index < len(driving_route):  # NOTE: the segment position index may end out of segments
        vehicle.store_fcd(dt, d, driving_route[segment_pos.index])

    # update the vehicle
    vehicle.time_offset += d
    vehicle.set_position(segment_pos)

    if vehicle.current_node == vehicle.dest_node:
        # stop the processing in case the ca    r reached the end
        vehicle.active = False

    return vehicle



