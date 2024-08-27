import logging
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple

from .kernels import AlternativesProvider, RouteSelectionProvider, VehicleWithPlans, AlternativeRoutes, \
    ZeroMQDistributedAlternatives, VehicleWithRoute
from .route import advance_vehicles_with_queues
from .simulation import FCDRecord, Simulation
from ..data.map import Map
from ..utils import TimerSet
from ..vehicle import Vehicle, VehicleAlternatives

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
            route_selection_providers: List[RouteSelectionProvider],
            end_step_fns: [Optional[Callable[[Simulation], None]]] = None,
    ):
        """Perform the simulation.

        Parameters:
        -----------
            :param alternatives_providers: Implementation of alternatives.
            :param route_selection_providers: Implementation of route selection.
            :param end_step_fns: An arbitrary functions that are called at the end of each step with
            the current state of simulation. It can be used for storing the state, for example.
        """

        for v in self.sim.vehicles:
            v.frequency = timedelta(seconds=5)

        self.sim.routing_map.update_temporary_max_speeds(self.sim.setting.departure_time + self.current_offset)

        for alternatives_provider in alternatives_providers:
            alternatives_provider.load_map(self.sim.routing_map)

            if self.sim.setting.plateau_default_route:
                if isinstance(alternatives_provider, ZeroMQDistributedAlternatives):
                    self.change_baseline_alternatives(self.sim.vehicles, alternatives_provider)

        step = self.sim.number_of_steps
        last_map_update = self.current_offset
        last_time_moved = self.current_offset
        updated_speeds = {}

        while self.current_offset is not None:
            step_start_dt = datetime.now()
            timer_set = TimerSet()

            offset = self.sim.round_time_offset(self.current_offset)

            if self.sim.setting.stuck_detection:
                # check if the simulation is stuck
                if ((self.current_offset - last_time_moved) >=
                        (self.sim.setting.round_freq * self.sim.setting.stuck_detection)):
                    logger.error(
                        f"The simulation is stuck at {self.current_offset}.")
                    break

            with timer_set.get("update_map_speeds"):
                last_map_update, updated_speeds = self.update_map_speeds(updated_speeds, last_map_update,
                                                                         alternatives_providers)

            with timer_set.get("allowed_vehicles"):
                vehicles_to_be_moved = [v for v in self.sim.vehicles
                                        if self.sim.is_vehicle_within_offset(v, offset)]

            with timer_set.get("alternatives"):
                need_new_route = [vehicle for vehicle in vehicles_to_be_moved if
                                  vehicle.is_at_the_end_of_segment(self.sim.routing_map)
                                  and vehicle.alternatives != VehicleAlternatives.DEFAULT]

                computed_vehicles, alts = compute_alternatives(alternatives_providers,
                                                               need_new_route, self.sim.setting.k_alternatives)
                assert len(computed_vehicles) == len(alts) == len(need_new_route)

            # Find which vehicles should have their routes recomputed
            with timer_set.get("collect"):
                new_vehicle_routes = []
                for v, alt in zip(computed_vehicles, alts):
                    if alt is not None and alt != []:
                        new_vehicle_routes.append((v, alt))

            with timer_set.get("selected_routes"):
                selected_plans = select_routes(route_selection_providers, new_vehicle_routes)
                assert len(selected_plans) == len(new_vehicle_routes)

                check_travel_times(self.sim.routing_map, alternatives_providers, selected_plans)

                current_map_id = self.sim.routing_map.map_id
                for (vehicle, route) in selected_plans:
                    vehicle.update_followup_route(route, self.sim.routing_map, self.sim.setting.travel_time_limit_perc)

            with timer_set.get("advance_vehicle"):
                fcds, has_moved = self.advance_vehicles(vehicles_to_be_moved.copy(), offset)
                if has_moved or self.sim.routing_map.has_temporary_speeds_planned():
                    last_time_moved = self.current_offset

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
                f"{step}. active: {len(vehicles_to_be_moved)}, need_new_route: {len(need_new_route)}, duration: {step_dur / timedelta(milliseconds=1)} ms, time: {self.current_offset}")
            self.sim.duration += step_dur

            if end_step_fns is not None:
                with timer_set.get("end_step"):
                    for fn in end_step_fns:
                        fn(self.state)

            self.sim.save_step_info(self.current_offset, step, len(vehicles_to_be_moved),
                                    step_dur, timer_set.collect(), len(need_new_route))

            step += 1
        logger.info(f"Simulation done in {self.sim.duration}.")

    def advance_vehicles(self, vehicles: List[Vehicle], current_offset) -> Tuple[List[FCDRecord], bool]:
        """Move the vehicles on its route and generate FCD records"""

        return advance_vehicles_with_queues(vehicles, self.sim.setting.departure_time,
                                            self.sim.global_view,
                                            self.sim.routing_map,
                                            self.sim.queues_manager,
                                            self.sim.setting.los_vehicles_tolerance)

    def change_baseline_alternatives(self,
                                     vehicles: List[Vehicle],
                                     alternatives_provider: AlternativesProvider):
        vehicles = [v for v in vehicles if v.osm_route is not None]
        logger.info(f"Computing default routes with {alternatives_provider.vehicle_behaviour}")
        alts = alternatives_provider.compute_alternatives(
            vehicles,
            k=1
        )
        assert len(vehicles) == len(alts)

        for vehicle, alt in zip(vehicles, alts):
            if alt is not None and alt != []:
                vehicle.update_followup_route(alt[0], self.sim.routing_map, travel_time_limit_perc=None)
            else:
                vehicle.osm_route = None
                vehicle.active = False
                vehicle.status = "no plateau route"

    def update_map_speeds(self, updated_speeds: dict, last_map_update: int,
                          alternatives_providers: List[AlternativesProvider]):
        # Get segments where max speeds changed
        updated_speeds.update(self.sim.routing_map.update_temporary_max_speeds(
            self.sim.setting.departure_time + self.current_offset))

        if self.current_offset - last_map_update >= self.sim.setting.map_update_freq_s:
            # Update speeds based on the global view
            updated_speeds_gv = self.sim.global_view.take_segment_speeds()
            updated_speeds.update(self.sim.routing_map.update_current_speeds(updated_speeds_gv))

            for alternatives_provider in alternatives_providers:
                alternatives_provider.update_map(updated_speeds)

            updated_speeds.clear()
            last_map_update = self.current_offset

        return last_map_update, updated_speeds


def compute_alternatives(alternatives_providers: List[AlternativesProvider],
                         vehicles: List[Vehicle],
                         k_alternatives: int) -> Tuple[List[Vehicle], List[AlternativeRoutes]]:
    if not vehicles:
        return [], []

    combined_alts = []
    combined_vehicles = []
    for provider in alternatives_providers:
        selected_vehicles = [v for v in vehicles if v.alternatives == provider.vehicle_behaviour]
        alts = provider.compute_alternatives(
            selected_vehicles,
            k=k_alternatives
        )
        combined_vehicles.extend(selected_vehicles)
        combined_alts.extend(alts)

    return combined_vehicles, combined_alts


def select_routes(route_selection_providers: List[RouteSelectionProvider], vehicles: List[VehicleWithPlans]):
    if not vehicles:
        return []

    combined_routes = []
    for provider in route_selection_providers:
        selected_vehicles = [(v, alt) for (v, alt) in vehicles if v.route_selection == provider.vehicle_behaviour]
        vehicles_with_route = provider.select_routes(selected_vehicles)
        combined_routes.extend(vehicles_with_route)

    assert len(combined_routes) == len(vehicles)
    return combined_routes


def check_travel_times(routing_map: Map, alternatives_providers: List[AlternativesProvider],
                       vehicles_with_routes: List[VehicleWithRoute]):

    current_map_id = routing_map.map_id
    vehicles_to_update = [v_r[0] for v_r in vehicles_with_routes if v_r[0].map_id != current_map_id
                          and v_r[0].alternatives not in [VehicleAlternatives.DEFAULT,
                                                          VehicleAlternatives.DIJKSTRA_SHORTEST]]

    for provider in alternatives_providers:
        current_routes = [v.osm_route for v in vehicles_to_update if v.alternatives == provider.vehicle_behaviour]
        travel_times = provider.get_routes_travel_times(current_routes)
        for vehicle, travel_time in zip(vehicles_to_update, travel_times):
            if travel_time is not None:
                vehicle.set_current_travel_time(travel_time, current_map_id)


def remove_infinity_alternatives(alternatives: List[AlternativeRoutes], routing_map: Map) \
        -> List[AlternativeRoutes]:
    """
    Removes alternatives that contain infinity.
    """
    filtered_alternatives = []
    for alternatives_for_vehicle in alternatives:
        for_vehicle = []
        for alternative in alternatives_for_vehicle:
            # calculate travel time for alternative
            if not routing_map.is_route_closed(alternative[0]):
                for_vehicle.append(alternative)
        filtered_alternatives.append(for_vehicle)
    return filtered_alternatives
