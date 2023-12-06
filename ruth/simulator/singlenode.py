import logging
from datetime import datetime, timedelta
from typing import Callable, List, Optional

from .kernels import AlternativeRoutes, AlternativesProvider, RouteSelectionProvider
from .route import advance_vehicles_with_queues
from .simulation import FCDRecord, Simulation
from ..data.map import Map
from ..losdb import GlobalViewDb
from ..utils import TimerSet
from ..vehicle import Vehicle, VehicleBehavior

logger = logging.getLogger(__name__)


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
            alternatives_providers: List[AlternativesProvider],
            route_selection_provider: RouteSelectionProvider,
            end_step_fns: [Optional[Callable[[Simulation], None]]] = None,
    ):
        """Perform the simulation.

        Parameters:
        -----------
            :param alternatives_providers: Implementation of alternatives.
            :param route_selection_provider: Implementation of route selection.
            :param end_step_fns: An arbitrary functions that are called at the end of each step with
            the current state of simulation. It can be used for storing the state, for example.
        """

        for v in self.sim.vehicles:
            v.frequency = timedelta(seconds=5)

        for alternatives_provider in alternatives_providers:
            alternatives_provider.load_map(self.sim.routing_map)

        step = self.sim.number_of_steps
        last_map_update = self.current_offset

        while self.current_offset is not None:
            step_start_dt = datetime.now()
            timer_set = TimerSet()

            offset = self.sim.round_time_offset(self.current_offset)

            with timer_set.get("update_map_speeds"):
                self.sim.routing_map.update_temporary_max_speeds(self.sim.setting.departure_time + self.current_offset)
                if self.current_offset - last_map_update >= self.sim.setting.map_update_freq_s:
                    updated_speeds = self.sim.global_view.take_segment_speeds()
                    self.sim.routing_map.update_current_speeds(updated_speeds)
                    for alternatives_provider in alternatives_providers:
                        alternatives_provider.update_map(self.sim.routing_map, updated_speeds)

                    last_map_update = self.current_offset

            with timer_set.get("allowed_vehicles"):
                vehicles_to_be_moved = [v for v in self.sim.vehicles
                                        if self.sim.is_vehicle_within_offset(v, offset)]

            with timer_set.get("alternatives"):
                need_new_route = [vehicle for vehicle in vehicles_to_be_moved if
                                  vehicle.is_at_the_end_of_segment(self.sim.routing_map)
                                  and vehicle.behavior != VehicleBehavior.DEFAULT]
                computed_vehicles, alts = self.compute_alternatives(alternatives_providers, need_new_route)
                assert len(computed_vehicles) == len(alts) == len(need_new_route)

            # Find which vehicles should have their routes recomputed
            with timer_set.get("collect"):
                new_vehicle_routes = []
                for v, alt in zip(computed_vehicles, alts):
                    if alt is not None and alt != []:
                        new_vehicle_routes.append((v, alt))

            with timer_set.get("selected_routes"):
                selected_plans = route_selection_provider.select_routes(new_vehicle_routes)
                assert len(selected_plans) == len(new_vehicle_routes)
                for (vehicle, route) in selected_plans:
                    vehicle.update_followup_route(route)

            with timer_set.get("advance_vehicle"):
                fcds = self.advance_vehicles(vehicles_to_be_moved.copy(), offset)

            with timer_set.get("update"):
                self.sim.update(fcds)

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
                f"{step}. active: {len(vehicles_to_be_moved)}, duration: {step_dur / timedelta(milliseconds=1)} ms, time: {self.current_offset}")
            self.sim.duration += step_dur

            if end_step_fns is not None:
                with timer_set.get("end_step"):
                    for fn in end_step_fns:
                        fn(self.state)

            self.sim.save_step_info(step, len(vehicles_to_be_moved), step_dur, timer_set.collect())

            step += 1
        logger.info(f"Simulation done in {self.sim.duration}.")

    def compute_alternatives(self, alternatives_providers: List[AlternativesProvider], vehicles: List[Vehicle]):
        if not vehicles:
            return [], []

        combined_alts = []
        combined_vehicles = []
        for provider in alternatives_providers:
            selected_vehicles = [v for v in vehicles if v.behavior == provider.vehicle_behaviour]
            alts = provider.compute_alternatives(
                selected_vehicles,
                k=self.sim.setting.k_alternatives
            )
            combined_vehicles.extend(selected_vehicles)
            combined_alts.extend(alts)

        combined_alts = remove_infinity_alternatives(combined_alts, self.sim.routing_map)
        return combined_vehicles, combined_alts

    def advance_vehicles(self, vehicles: List[Vehicle], current_offset) -> List[FCDRecord]:
        """Move the vehicles on its route and generate FCD records"""

        for vehicle in vehicles:
            assert vehicle.is_active(current_offset, self.sim.setting.round_freq)
        return advance_vehicles_with_queues(vehicles, self.sim.setting.departure_time,
                                            GlobalViewDb(self.sim.global_view),
                                            self.sim.routing_map,
                                            self.sim.queues_manager,
                                            self.sim.setting.los_vehicles_tolerance)


def remove_infinity_alternatives(alternatives: List[AlternativeRoutes], routing_map: Map) -> List[
    AlternativeRoutes]:
    """
    Removes alternatives that contain infinity.
    """
    filtered_alternatives = []
    for alternatives_for_vehicle in alternatives:
        for_vehicle = []
        for alternative in alternatives_for_vehicle:
            # calculate travel time for alternative
            if not routing_map.is_route_closed(alternative):
                for_vehicle.append(alternative)
        filtered_alternatives.append(for_vehicle)
    return filtered_alternatives
