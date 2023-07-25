import logging
from datetime import datetime, timedelta
from typing import Callable, Optional

from .kernels import AlternativesProvider, RouteSelectionProvider, VehicleWithRoute
from .route import advance_vehicle
from .simulation import Simulation, VehicleUpdate
from ..losdb import GlobalViewDb
from ..utils import TimerSet
from ..vehicle import Vehicle

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

            # Find which vehicles should have their routes recomputed
            with timer_set.get("collect"):
                need_new_route = []
                for v, alt in zip(allowed_vehicles, alts):
                    if alt is not None and alt != []:
                        need_new_route.append((v, alt))

            with timer_set.get("selected_routes"):
                selected_plans = route_selection_provider.select_routes(need_new_route)
                assert len(selected_plans) == len(need_new_route)
                for (vehicle, route) in selected_plans:
                    vehicle.update_followup_route(route)

            with timer_set.get("advance_vehicle"):
                vehicle_updates = [self.advance_vehicle(vehicle, offset) for vehicle in allowed_vehicles]

            with timer_set.get("update"):
                self.sim.update(vehicle_updates)

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

            # print(timer_set.collect())

            self.sim.save_step_info(step, len(allowed_vehicles), step_dur, timer_set.collect())

            step += 1
        logger.info(f"Simulation done in {self.sim.duration}.")

    def advance_vehicle(self, vehicle: Vehicle, current_offset) -> VehicleUpdate:
        """Move with the vehicle on the route (update its state), and disentangle its leap history"""

        leap_history = []
        if vehicle.is_active(current_offset, self.sim.setting.round_freq):
            advance_vehicle(vehicle, self.sim.setting.departure_time,
                            GlobalViewDb(self.sim.global_view))
            # swap the empty history with the filled one
            leap_history, vehicle.leap_history = vehicle.leap_history, leap_history

        return VehicleUpdate(vehicle, leap_history)
