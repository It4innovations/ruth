import functools
import logging

from datetime import datetime, timedelta
from itertools import chain, groupby
from multiprocessing import Pool
from typing import Any, List, Dict, Tuple, Callable, NewType

import pylru
from probduration import VehiclePlan, Route, SegmentPosition


from .common import alternatives, advance_vehicle
from .routeranking import Comparable
from ..globalview import GlobalView
from ..utils import osm_route_to_segments, route_to_osm_route, TimerSet
from ..vehicle import Vehicle
from ..losdb import GlobalViewDb

from .simulation import Simulation, VehicleUpdate


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


VehiclePlans = NewType("VehiclePlans", List[Tuple[Vehicle, VehiclePlan]])


class Simulator:

    def __init__(self, sim: Simulation, nproc=1):
        """Initialize the simulator.

        Parameters:
        -----------
            sim: Simulation
                State of the simulation.
            nproc: int [default=1]
                Number of concurrent processes.
        """
        self.sim = sim
        self.pool = None
        self.nproc = nproc
        self.current_offset = self.sim.compute_current_offset()
        self.alternatives_cache = pylru.lrucache(100_000)

    def __enter__(self):
        self.pool = Pool(processes=self.nproc)
        return self

    def __exit__(self, exc_type_, exc_val_, exc_tb_):
        self.pool.close()

    @property
    def state(self):
        return self.sim

    def simulate(self,
                 route_ranking_fn: Callable[[GlobalView, VehiclePlans, List, Dict], Comparable],
                 rr_fn_args=(),
                 rr_fn_kwargs=None,
                 extend_plans_fn: Callable[[VehiclePlans, List, Dict], Any] = lambda plans: plans,
                 ep_fn_args=(),
                 ep_fn_kwargs=None,
                 end_step_fn: Callable[[Simulation, List, Dict], None] = lambda *_: None,
                 es_fn_args=(),
                 es_fn_kwargs=None):
        """Perform the simulation.

        Parameters:
        -----------
            route_ranking_fn: Callable[[GlobalView, VehiclePlans, List, Dict], Comparable]
              Compute a (comparable) rank for a route
            rr_fn_args: Tuple
              Positional arguments for route ranking function
            rr_fn_kwargs: Dict
              Keyword arguments for route ranking function
            extend_plans_fn: Callable[[VehiclePlans, List, Dict, Any]
              Extends a list of vehicle plans about additional information. This information can be then accessed
              within route ranking function.
            ep_fn_args: Tuple
              Positional arguments for extend vehicle plans function
            ep_fn_kwargs: Dict
              Keyword arguments for extend vehicle plans function
            end_step_fn: Callable[[Simulation], None]
              An arbitrary function that is called at the end of each step with the current state of simulation.
              It can be used for storing the state, for example.
            es_fn_args: Tuple
              Positional arguments for end-step function.
            es_fn_kwargs: Dict
              Keyword arguments for end-step function
        """

        step = self.sim.number_of_steps
        while self.current_offset is not None:
            step_start_dt = datetime.now()
            timer_set = TimerSet()

            offset = self.sim.round_time_offset(self.current_offset)

            with timer_set.get("allowed_vehicles"):
                allowed_vehicles = [v for v in self.sim.vehicles
                                    if self.sim.is_vehicle_within_offset(v, offset)]

            with timer_set.get("alternatives"):
                alts = self.alternatives(allowed_vehicles)
            alt_dict = dict((v.id, (v, alt)) for v, alt in alts)

            # collect vehicles without alternative and finish them
            with timer_set.get("collect"):
                not_moved = []
                for v in allowed_vehicles:
                    if v.id not in alt_dict:
                        v.active = False
                        v.status = "no-route-exists"
                        leap_history, v.leap_history = v.leap_history, []
                        not_moved.append(VehicleUpdate(v, leap_history))

            with timer_set.get("vehicle_plans"):
                vehicle_plans = chain.from_iterable(
                    filter(None, map(functools.partial(prepare_vehicle_plans,
                                                       departure_time=self.sim.setting.departure_time),
                                     alts)))

            with timer_set.get("select_plans"):
                selected_plans = select_plans(vehicle_plans,
                                              route_ranking_fn, rr_fn_args, rr_fn_kwargs,
                                              extend_plans_fn, ep_fn_args, ep_fn_kwargs)

            assert selected_plans, "Unexpected empty list of selected plans."

            def transform_plan(vehicle_plan):
                vehicle, plan = vehicle_plan
                return vehicle, route_to_osm_route(plan.route)

            with timer_set.get("transform_plans"):
                bests = list(map(transform_plan, selected_plans))

            with timer_set.get("advance_vehicle"):
                new_vehicles = [self.advance_vehicle(best, offset) for best in bests] + not_moved

            with timer_set.get("update"):
                self.sim.update(new_vehicles)

            with timer_set.get("compute_offset"):
                current_offset_new = self.sim.compute_current_offset()
                if current_offset_new == self.current_offset:
                    logger.error(f"The consecutive step with the same offset: {self.current_offset}.")
                    break
                self.current_offset = current_offset_new

            with timer_set.get("drop_old_records"):
                self.sim.drop_old_records(self.current_offset)

            step_dur = datetime.now() - step_start_dt
            logger.info(f"{step}. active: {len(allowed_vehicles)} duration: {step_dur / timedelta(milliseconds=1)} ms")
            self.sim.duration += step_dur

            with timer_set.get("end_step"):
                es_fn_kwargs_ = {} if es_fn_kwargs is None else es_fn_kwargs
                end_step_fn(self, *es_fn_args, **es_fn_kwargs_)

            self.sim.save_step_info(step, len(allowed_vehicles), step_dur, timer_set.collect())

            step += 1
        logger.info(f"Simulation done in {self.sim.duration}.")

    def alternatives(self, vehicles):
        if self.pool is None:
            logger.info("The alternative routes are computed without multiprocessing.")
            map_fn = map
        else:
            map_fn = self.pool.map

        by_id = {}

        hits = 0
        cached_vehicles = []
        uncached_vehicles = []
        for (index, vehicle) in enumerate(vehicles):
            by_id[vehicle.id] = index
            od = vehicle.current_od
            result = self.alternatives_cache.get(od)
            if result is not None:
                cached_vehicles.append((vehicle, result))
                hits += 1
            else:
                uncached_vehicles.append(vehicle)

        logger.info(f"Alternatives hit rate: {hits}/{len(vehicles)} ({(hits / len(vehicles)) * 100:.2f}%)")

        alts = list(filter(None, map_fn(functools.partial(alternatives, k=self.sim.setting.k_alternatives),
                           uncached_vehicles)))
        for (vehicle, result) in alts:
            self.alternatives_cache[vehicle.current_od] = result

        alts += cached_vehicles
        alts = sorted(alts, key=lambda item: by_id[item[0].id])
        alts = [(
            vehicle,
            [vehicle.osm_route[:vehicle.next_routing_start_node_with_index[1]] + osm_route for osm_route in osm_routes]
        ) for (vehicle, osm_routes) in alts]

        if not alts:
            offsets = sorted(v.time_offset for v in vehicles)
            logger.debug(f"No alternatives found at offset range: ({offsets[0]}, {offsets[-1]})")
        return alts

    def advance_vehicle(self, vehicle_route, current_offset):
        """Move with the vehicle on the route (update its state), and disentangle its leap history"""

        vehicle, osm_route = vehicle_route

        leap_history = []
        if vehicle.is_active(current_offset, self.sim.setting.round_freq):
            new_vehicle = advance_vehicle(vehicle, osm_route,
                                          self.sim.setting.departure_time, GlobalViewDb(self.sim.global_view))
            # swap the empty history with the filled one
            leap_history, new_vehicle.leap_history = new_vehicle.leap_history, leap_history
        else:
            new_vehicle = vehicle

        return VehicleUpdate(new_vehicle, leap_history)


def select_plans(vehicle_plans,
                 rank_fn, rank_fn_args=(), rank_fn_kwargs=None,
                 extend_plans_fn=lambda plans: plans, ep_fn_args=(), ep_fn_kwargs=None):
    """Select from alternatives based on rank function.

    Parameters:
    -----------
        vehicle_plans: List[Tuple[Vehicle, VehiclePlan]]
        rank_fn: Callable[[List, Dict], Comparable]
          A route ranking function
        rank_fn_args: Tuple
        rank_fn_kwargs: Dict
        extend_plans_fn: Callable[[List, Dict], Comparable]
          A function that can extend the vehicle plans by additional information which can be used in route ranking
          function.
        ep_fn_args: Tuple
        ep_fn_kwargs: Dict
    """

    vehicle_plans_ = list(vehicle_plans)  # materialize the iterator as it is used twice

    ep_fn_kwargs_ = {} if ep_fn_kwargs is None else ep_fn_kwargs
    vehicle_plans_extended = extend_plans_fn(vehicle_plans_, *ep_fn_args, **ep_fn_kwargs_)

    rank_fn_kwargs_ = {} if rank_fn_kwargs is None else rank_fn_kwargs
    ranks = map(functools.partial(rank_fn, *rank_fn_args, **rank_fn_kwargs_), vehicle_plans_extended)

    def by_plan_id(data):
        (_, plan), _ = data
        return plan.id

    def by_rank(data):
        _, rank = data
        return rank

    data = groupby(sorted(zip(vehicle_plans_, ranks), key=by_plan_id), by_plan_id)

    def select_best(data):
        plan_id, group = data
        # a single group contains tuple with vehicle plan and computed duration, respectively
        return sorted(group, key=by_rank)[0][0]

    return list(map(select_best, data))


def prepare_vehicle_plans(alt, departure_time):
    """Prepare a list of vehicle plans for particular alternative. Returns a list of pairs: (`Vehicle`, `VehiclePlan`)

    Parameters:
    -----------
        alt: (Vehicle, List[int])
          A pair of vehicle and list of alternative osm routes (one osm route is a list of node ids).
        departure_time: datetime
          A departure time of the simulation.

    """
    vehicle, osm_routes = alt
    if osm_routes is None:
        logger.debug(f"No alternative for vehicle: {vehicle.id}")
        return None

    return [(vehicle, VehiclePlan(vehicle.id,
                                  Route(osm_route_to_segments(osm_route, vehicle.routing_map), vehicle.frequency),
                                  SegmentPosition(0, 0.0),
                                  departure_time + vehicle.time_offset))
            for osm_route in osm_routes]
