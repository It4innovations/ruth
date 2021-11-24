"""A distributed traffic simulator."""

import os
import random
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from dataclasses import fields

from probduration import HistoryHandler, Route, probable_duration

from ruth.utils import osm_route_to_segments
from ruth.vehicle import Vehicle
from ruth.pandasdataclasses import DataFrameRow


def simulate(input_csv,
             n_workers,
             departure_time,
             k_routes,
             n_samples,
             seed,
             gv_update_period,
             dask_scheduler,
             dask_scheduler_port,
             intermediate_results,
             checkpoint_period):
    """Distributed traffic simulator."""

    if seed is not None:
        random.seed(seed)  # used for tests: 660277

    if intermediate_results is not None:
        intermediate_results = os.path.abspath(intermediate_results)

        if not os.path.exists(intermediate_results):
            os.mkdir(intermediate_results)

    c = Client(f"{dask_scheduler}:{dask_scheduler_port}")

    df = pd.read_pickle(input_csv)

    # TODO: get types for meta
    types = dict(map(lambda field: (field.name, field.type), fields(Vehicle)))
    affected_columns = list(types.keys())

    round = 0
    active = True
    while active:
        ddf = dd.from_pandas(df, npartitions=n_workers)
        ddf = c.persist(ddf)
        for _ in range(gv_update_period):
            min_offset = ddf["time_offset"].min()

            cond = (ddf["time_offset"] == min_offset) & (ddf["active"])

            new_values = ddf.loc[cond, affected_columns].apply(
                advance_vehicle,
                axis=1,
                args=(n_samples, k_routes, departure_time),
                meta={  # TODO: solve better the meta types
                    'id': 'int64',
                    'time_offset': 'object',  # `object` as it is `timedelta` type
                    'frequency': 'object',  # `object` as it is `timedelta` type
                    'start_index': 'int64',
                    'start_distance_offset': 'float64',
                    'origin_node': 'int64',
                    'dest_node': 'int64',
                    'border_id': 'string',
                    'osm_route': 'object',
                    'active': 'bool'})  # NOTE: the types are important especially for objectts!

            ddf[affected_columns] = ddf[affected_columns].mask(cond, new_values)
        df = ddf.compute()
        # store intermediate results if desired
        if intermediate_results is not None and round % checkpoint_period == 0:
            df.to_pickle(f"{intermediate_results}/df_{round + 1}.pickle")
        round += 1
        active = df["active"].any()

    return df


@DataFrameRow(Vehicle)
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

    # update the vehicle
    vehicle.time_offset += d
    vehicle.set_position(segment_pos)

    if vehicle.current_node == vehicle.dest_node:
        # stop the processing in case the car reached the end
        vehicle.active = False

    return vehicle
