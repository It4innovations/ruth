import functools
import logging
from datetime import datetime, timedelta
from itertools import groupby
from typing import Callable, List, NewType, Optional, Tuple

from probduration import Route, SegmentPosition, VehiclePlan

from .common import advance_vehicle
from .kernels import AlternativesProvider, RouteSelectionProvider, VehicleWithRoute
from .simulation import Simulation, VehicleUpdate
from ..losdb import GlobalViewDb
from ..utils import TimerSet, osm_route_to_segments
from ..vehicle import Vehicle

logger = logging.getLogger(__name__)

VehiclePlans = NewType("VehiclePlans", List[Tuple[Vehicle, VehiclePlan]])


class Simulator:

    def __init__(self, sim: Simulation):
        """Initialize the simulator.

        Parameters:
        -----------
            sim: Simulation
                State of the simulation.
        """
        self.sim = sim
        self.current_offset = self.sim.compute_current_offset()

    @property
    def state(self):
        return self.sim

    def simulate(
            self,
            alternatives_provider: AlternativesProvider,
            route_selection_provider: RouteSelectionProvider,
            end_step_fn: Optional[Callable[[Simulation], None]] = None,
    ):
        """Perform the simulation.

        Parameters:
        -----------
            :param alternatives_provider: Implementation of alternatives.
            :param route_selection_provider: Implementation of route selection.
            :param end_step_fn: An arbitrary function that is called at the end of each step with
            the current state of simulation. It can be used for storing the state, for example.
        """

        for v in self.sim.vehicles:
            v.frequency = timedelta(seconds=5)

        step = self.sim.number_of_steps
        while self.current_offset is not None:
            step_start_dt = datetime.now()
            timer_set = TimerSet()

            offset = self.sim.round_time_offset(self.current_offset)

            with timer_set.get("allowed_vehicles"):
                allowed_vehicles = [v for v in self.sim.vehicles
                                    if self.sim.is_vehicle_within_offset(v, offset)]

            with timer_set.get("alternatives"):
                alts = alternatives_provider.compute_alternatives(
                    allowed_vehicles,
                    k=self.sim.setting.k_alternatives
                )
                assert len(alts) == len(allowed_vehicles)

            # collect vehicles without alternative and finish them
            with timer_set.get("collect"):
                not_moved = []
                moving = []
                for v, alt in zip(allowed_vehicles, alts):
                    if alt is None:
                        v.active = False
                        v.status = "no-route-exists"
                        leap_history, v.leap_history = v.leap_history, []
                        not_moved.append(VehicleUpdate(v, leap_history))
                    else:
                        moving.append((v, alt))

            with timer_set.get("selected_routes"):
                selected_plans = route_selection_provider.select_routes(moving)
                assert len(selected_plans) == len(moving)

            with timer_set.get("advance_vehicle"):
                new_vehicles = [self.advance_vehicle(plan, offset) for plan in
                                selected_plans] + not_moved

            with timer_set.get("update"):
                self.sim.update(new_vehicles)

            with timer_set.get("compute_offset"):
                current_offset_new = self.sim.compute_current_offset()
                if current_offset_new == self.current_offset:
                    logger.error(
                        f"The consecutive step with the same offset: {self.current_offset}.")
                    break
                self.current_offset = current_offset_new

            with timer_set.get("drop_old_records"):
                self.sim.drop_old_records(self.current_offset)

            step_dur = datetime.now() - step_start_dt
            logger.info(
                f"{step}. active: {len(allowed_vehicles)}, duration: {step_dur / timedelta(milliseconds=1)} ms")
            self.sim.duration += step_dur

            if end_step_fn is not None:
                with timer_set.get("end_step"):
                    end_step_fn(self.state)

            self.sim.save_step_info(step, len(allowed_vehicles), step_dur, timer_set.collect())

            step += 1
        logger.info(f"Simulation done in {self.sim.duration}.")

    def advance_vehicle(self, vehicle_route: VehicleWithRoute, current_offset):
        """Move with the vehicle on the route (update its state), and disentangle its leap history"""

        vehicle, osm_route = vehicle_route

        leap_history = []
        if vehicle.is_active(current_offset, self.sim.setting.round_freq):
            new_vehicle = advance_vehicle(vehicle, osm_route,
                                          self.sim.setting.departure_time,
                                          GlobalViewDb(self.sim.global_view))
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
    ranks = map(functools.partial(rank_fn, *rank_fn_args, **rank_fn_kwargs_),
                vehicle_plans_extended)

    def by_plan_id(data):
        (_, plan), _ = data
        return plan.id

    def by_rank(data):
        _, rank = data
        return rank

    grouped_data = groupby(sorted(zip(vehicle_plans_, ranks), key=by_plan_id), by_plan_id)

    def select_best(data):
        plan_id, group = data
        # a single group contains tuple with vehicle plan and computed duration, respectively
        group = list(group)
        prev_plan, prev_rank = group[0]  # at the first position is the path from previous step
        best_plan, best_rank = sorted(group, key=by_rank)[0]
        if abs(prev_rank - best_rank) > timedelta(
                minutes=1):  # TODO: expose 1m epsilon to the user interface
            # TODO: collect hits: "switch from previous to the best"
            return best_plan
        return prev_plan

    return list(map(select_best, grouped_data))


def prepare_vehicle_plans(vehicle_alts, departure_time):
    """Prepare a list of vehicle plans for particular alternative. Returns a list of pairs: (`Vehicle`, `VehiclePlan`)

    Parameters:
    -----------
        vehicle_alts: (Vehicle, List[int])
          A pair of vehicle and list of alternative osm routes (one osm route is a list of node ids).
        departure_time: datetime
          A departure time of the simulation.

    """
    vehicle, osm_routes = vehicle_alts
    if osm_routes is None:
        logger.debug(f"No alternative for vehicle: {vehicle.id}")
        return None

    return [(vehicle, VehiclePlan(vehicle.id,
                                  Route(osm_route_to_segments(osm_route, vehicle.routing_map),
                                        vehicle.frequency),
                                  SegmentPosition(0, 0.0),
                                  departure_time + vehicle.time_offset))
            for osm_route in osm_routes]
