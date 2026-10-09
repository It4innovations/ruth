import logging
import os
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple

from .kernels import AlternativesProvider, RouteSelectionProvider, VehicleWithPlans, AlternativeRoutes, \
    VehicleWithRoute, MPIDistributedAlternatives
from .route import advance_vehicles_with_queues
from .simulation import FCDRecord, Simulation
from ..data.map import Map
from ..utils import TimerSet
from ..vehicle import Vehicle, VehicleAlternatives

logger = logging.getLogger(__name__)
PLATEAU_DEFAULT_ROUTE_SPEEDS_ENV = "RUTH_PLATEAU_DEFAULT_ROUTE_SPEEDS"


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
            end_step_fns: Optional[List[Callable[[Simulation], bool]]] = None,
            max_steps: Optional[int] = None,
    ):
        """Perform the simulation.

        Parameters:
        -----------
            :param alternatives_providers: Implementation of alternatives.
            :param route_selection_providers: Implementation of route selection.
            :param end_step_fns: Functions called at the end of each step with the current state.
            Returning True requests a graceful stop after the current step.
            :param max_steps: Stop after this many total steps. Steps from a resumed simulation
            count toward the limit. None runs the simulation to completion.
        """
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        for alternatives_provider in alternatives_providers:
            alternatives_provider.load_map(self.sim.routing_map)

        if self.sim.streaming:
            self.prepare_loaded_vehicles(self.sim.vehicles, alternatives_providers)
            self.sim.prune_inactive_vehicles()
            self.load_due_streaming_vehicles(alternatives_providers)
            self.current_offset = self.sim.compute_current_offset()

        if self.current_offset is None:
            logger.warning("No active vehicles found. Simulation cannot proceed.")
            return

        step = self.sim.number_of_steps
        if max_steps is not None and step >= max_steps:
            logger.info("Simulation already reached the configured limit of %s steps.", max_steps)
            return

        self.sim.routing_map.update_temporary_max_speeds(self.sim.setting.departure_time + self.current_offset)

        moved_last_step = True
        last_map_update = self.current_offset
        last_time_moved = self.current_offset
        updated_speeds = {}

        if self.sim.last_saved_speeds:
            self.sim.routing_map.update_current_speeds(self.sim.last_saved_speeds)

        if step == 0 and not self.sim.streaming:
            for alternatives_provider in alternatives_providers:
                if self.sim.setting.plateau_default_route and isinstance(alternatives_provider, MPIDistributedAlternatives):
                    self.change_baseline_alternatives(self.sim.vehicles, alternatives_provider)

        last_saved_hour = -1
        with self.sim.history:
            # save the initial map to hdf5
            self.sim.history.writer.save_map(self.sim.routing_map, self.sim.setting.departure_time, self.sim.setting.round_freq)

            while self.current_offset is not None and (max_steps is None or step < max_steps):
                step_start_dt = datetime.now()
                timer_set = TimerSet()

                offset, offset_seconds = self.sim.round_time_offset(self.current_offset)

                if self.sim.setting.stuck_detection:
                    # check if the simulation is stuck
                    if ((self.current_offset - last_time_moved) >=
                            (self.sim.setting.round_freq * self.sim.setting.stuck_detection)):
                        if not moved_last_step:
                            logger.error(
                                f"The simulation is stuck at {self.current_offset}.")
                            break
                # END if self.sim.setting.stuck_detection

                with timer_set.get("store_sync"):
                    self.sync_store_population()

                with timer_set.get("update_map_speeds"):
                    last_map_update, updated_speeds = self.update_map_speeds(updated_speeds, last_map_update,
                                                                             alternatives_providers)

                with timer_set.get("allowed_vehicles"):
                    vehicles_to_be_moved = [v for v in self.sim.vehicles
                                            if self.sim.is_vehicle_within_offset(v, offset_seconds)]

                    need_new_route = [vehicle for vehicle in vehicles_to_be_moved if
                                      vehicle.is_at_the_end_of_segment(self.sim.routing_map)
                                      and vehicle.alternatives != VehicleAlternatives.DEFAULT]

                self.shadow_check_selection(offset_seconds, vehicles_to_be_moved)

                with timer_set.get("alternatives"):
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

                    for (vehicle, route) in selected_plans:
                        vehicle.update_followup_route(route, self.sim.routing_map,
                                                      self.sim.setting.travel_time_limit_perc)

                    # The store holds each vehicle's route; these just changed.
                    self.push_store_routes([v for (v, _) in selected_plans])

                with timer_set.get("advance_vehicle"):
                    fcds, has_moved = self.advance_vehicles(vehicles_to_be_moved.copy())
                    moved_last_step = False
                    if has_moved or self.sim.routing_map.has_temporary_speeds_planned():
                        last_time_moved = self.current_offset
                        moved_last_step = True
                # Post-advance state, before pruning changes the population: this
                # is the point both the serial and the distributed path must agree
                # on, step for step.
                self.emit_trajectory_digest(step)

                with timer_set.get("update_global_view"):
                    self.sim.update(fcds)

                with timer_set.get("fcd_history_extend"):
                    self.sim.history.extend(fcds)

                with timer_set.get("compute_offset"):
                    previous_offset = self.current_offset
                    self.sim.prune_inactive_vehicles()
                    current_offset_new = self.sim.compute_current_offset()
                    if self.sim.streaming:
                        self.load_due_streaming_vehicles(alternatives_providers)
                        current_offset_new = self.sim.compute_current_offset()
                    if current_offset_new == previous_offset:
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

                parts = timer_set.collect()
                parts.update(self.sim.history.collect_step_metrics(reset=True))

                logger.info(
                    "step_timers: step=%s %s",
                    step,
                    " ".join(f"{key}={value:.3f}" for key, value in sorted(parts.items()))
                )

                self.sim.save_step_info(self.current_offset, step, len(vehicles_to_be_moved),
                                        step_dur, parts, len(need_new_route))
                step += 1

                if end_step_fns is not None:
                    with timer_set.get("end_step"):
                        stop_requested = False
                        for fn in end_step_fns:
                            stop_requested = fn(self.state) or stop_requested

                if end_step_fns is not None and stop_requested:
                    logger.info("Simulation stopped by an end-of-step callback.")
                    break
            # END while self.current_offset is not None

            self.sim.history.writer.save_computational_time(self.sim.duration.total_seconds())
            if max_steps is not None and step >= max_steps and self.current_offset is not None:
                logger.info("Simulation stopped after reaching the configured limit of %s steps.", max_steps)
            logger.info(f"Simulation done in {self.sim.duration}.")

    def emit_trajectory_digest(self, step: int):
        """Log one hash of the post-advance movement state (ace.trajectory-digest).

        This is the compliance check for "the vehicles are in the same exact
        position": diff the logged digest column between a serial run and a
        distributed run at any rank count, and the first differing line is the
        first diverging step. It deliberately does not look at the FCD history,
        whose row order is allowed to differ.

        Off by default — it costs one O(N) pass per step.
        """
        setting = self.sim.setting
        # getattr throughout: see the note in advance_vehicles about checkpoints
        # written before these settings existed.
        if not getattr(setting, "trajectory_digest", False):
            return

        # Imported lazily so the module is not a hard dependency of a normal run.
        from .trajectory_digest import dump_trajectory, trajectory_digest

        logger.info("trajectory_digest: step=%s n=%s digest=%s",
                    step, len(self.sim.vehicles), trajectory_digest(self.sim.vehicles))

        dump_step = getattr(setting, "trajectory_dump_step", -1)
        if dump_step is not None and dump_step >= 0 and step == dump_step:
            path = f"trajectory-step{step:06d}.csv"
            rows = dump_trajectory(self.sim.vehicles, path)
            logger.info("trajectory_digest: wrote %s rows of step %s state to %s",
                        rows, step, path)

    def _use_store(self) -> bool:
        """Whether the C++ VehicleStore is in use this run.

        True when it is producing the tasks, and also when only the shadow check
        is on -- the check reads the live store, so the store has to be kept
        current either way.
        """
        setting = self.sim.setting
        if not getattr(setting, "mpi_advance", False):
            return False
        return (getattr(setting, "store_tasks", True)
                or getattr(setting, "store_shadow_check", False))

    def sync_store_population(self):
        """Re-initialise the store if the vehicle population changed.

        Streaming adds vehicles and pruning removes them, and the store is keyed
        by vehicle id, so a population change has to be noticed before anything
        asks it for a task. Comparing the id vectors is cheap and, unlike a
        dirty flag maintained at every call site that mutates the list, cannot
        be forgotten.
        """
        if not self._use_store():
            return

        from .store_bridge import ensure_population
        if ensure_population(self.sim.vehicles, self.sim.routing_map):
            logger.info("store: re-initialised for %d vehicles", len(self.sim.vehicles))

    def push_store_routes(self, vehicles: List[Vehicle]):
        """Push rewritten routes into the store."""
        if not self._use_store() or not vehicles:
            return

        from .store_bridge import push_routes
        push_routes(vehicles, self.sim.routing_map)

    def push_store_closures(self, segment_keys):
        """Push changed segment closures into the store.

        `is_closed` in a task mirrors `get_current_max_speed(...) == 0`, and only
        Map.update_temporary_max_speeds writes the field behind it. Missing an
        update here would leave a vehicle driving through a closed segment, so
        it is fed straight from that call's return value.
        """
        if not self._use_store() or not segment_keys:
            return

        from .store_bridge import push_closures
        push_closures(segment_keys, self.sim.routing_map)

    def shadow_check_selection(self, offset_seconds: int, selected: List[Vehicle]):
        """Check the C++ VehicleStore reproduces this step's selection.

        Verification only (ace.store-shadow-check); nothing here feeds the
        simulation. Unlike the task check in mpi_advance, this runs on the serial
        path too, because the selection it verifies is not specific to the
        distributed advance.

        It populates the store separately from the task check rather than
        sharing one population: update_followup_route rewrites routes between
        this point and task construction, so a population made here would be
        stale by then.
        """
        if not getattr(self.sim.setting, "store_shadow_check", False):
            return

        from .store_bridge import push_movement, shadow_check_selection

        # Selection looks at every vehicle, not just the moving ones, so the
        # whole population's movement state has to be current. In production only
        # the moving subset is pushed (in mpi_advance), which is all a task needs.
        push_movement(self.sim.vehicles)

        # The geometric half of need_new_route on its own; the alternatives mode
        # is a Python-only attribute the store does not carry.
        at_segment_end = [vehicle for vehicle in selected
                          if vehicle.is_at_the_end_of_segment(self.sim.routing_map)]

        shadow_check_selection(self.sim.vehicles, self.sim.routing_map,
                               offset_seconds, self.sim._freq_seconds,
                               selected, at_segment_end,
                               self.sim.compute_current_offset())

    def _use_mpi_advance(self) -> bool:
        # Driven by the config.json "ace" section (ace.mpi-advance), carried on the
        # SimSetting. Previously toggled via the RUTH_MPI_ADVANCE env var.
        return self.sim.setting.mpi_advance

    def advance_vehicles(self, vehicles: List[Vehicle]) -> Tuple[List[FCDRecord], bool]:
        """Move the vehicles on its route and generate FCD records"""

        if self._use_mpi_advance():
            # LoS is evaluated against the C++ GlobalView replicas (kept in sync
            # by the all-gather), so the Python global_view is not passed here.
            from .mpi_advance import advance_vehicles_mpi
            drop_before = (self.sim.setting.departure_time + self.current_offset
                           if self.current_offset is not None else None)
            # getattr: a simulation resumed from a checkpoint written before this
            # setting existed unpickles a SimSetting without the field (dataclass
            # defaults do not apply to __setstate__), and must keep running.
            return advance_vehicles_mpi(vehicles, self.sim.setting.departure_time,
                                        self.sim.routing_map,
                                        self.sim.queues_manager,
                                        self.sim.setting.los_vehicles_tolerance,
                                        drop_before_time=drop_before,
                                        deterministic_order=getattr(
                                            self.sim.setting, "deterministic_order", False),
                                        store_shadow_check=getattr(
                                            self.sim.setting, "store_shadow_check", False),
                                        store_tasks=getattr(
                                            self.sim.setting, "store_tasks", True))

        return advance_vehicles_with_queues(vehicles, self.sim.setting.departure_time,
                                            self.sim.global_view,
                                            self.sim.routing_map,
                                            self.sim.queues_manager,
                                            self.sim.setting.los_vehicles_tolerance)

    def change_baseline_alternatives(self,
                                     vehicles: List[Vehicle],
                                     alternatives_provider: AlternativesProvider):
        vehicles = [v for v in vehicles if v.osm_route is not None]
        use_origin_speeds = not self.plateau_default_route_uses_current_speeds()
        speed_mode = "max" if use_origin_speeds else "current"
        logger.info(f"Computing default routes with {alternatives_provider.vehicle_behaviour} using {speed_mode} speeds")
        alts = alternatives_provider.compute_alternatives(
            vehicles,
            k=1,
            use_origin_speeds=use_origin_speeds
        )

        assert len(vehicles) == len(alts)

        for vehicle, alt in zip(vehicles, alts):
            if alt is not None and alt != []:
                vehicle.update_followup_route(alt[0], self.sim.routing_map, travel_time_limit_perc=None)
            else:
                vehicle.osm_route = []
                vehicle.active = False
                vehicle.status = "no plateau route"

    def prepare_loaded_vehicles(self, vehicles: List[Vehicle],
                                alternatives_providers: List[AlternativesProvider]):
        vehicles = [v for v in vehicles if not getattr(v, "streaming_prepared", False)]
        if not vehicles:
            return

        routed_vehicles = [v for v in vehicles if not getattr(v, "needs_default_route", False)]
        if routed_vehicles:
            self.sim.routing_map.fix_osm_routes(routed_vehicles)

        unrouted_vehicles = [
            v for v in vehicles
            if v.active and getattr(v, "needs_default_route", False)
        ]
        if unrouted_vehicles:
            plateau_provider = next(
                (provider for provider in alternatives_providers
                 if isinstance(provider, MPIDistributedAlternatives)),
                None,
            )
            if plateau_provider is None:
                logger.error("Cannot compute default Plateau routes for %d streamed vehicles: Plateau provider missing.",
                             len(unrouted_vehicles))
                for vehicle in unrouted_vehicles:
                    vehicle.active = False
                    vehicle.status = "no plateau provider for default route"
            else:
                self.change_baseline_alternatives(unrouted_vehicles, plateau_provider)

        for vehicle in vehicles:
            vehicle.streaming_prepared = True
            vehicle.needs_default_route = False

    def load_due_streaming_vehicles(self, alternatives_providers: List[AlternativesProvider]):
        while self.sim.should_load_next_vehicle_bucket():
            vehicles = self.sim.load_next_vehicle_bucket()
            self.prepare_loaded_vehicles(vehicles, alternatives_providers)
            self.sim.prune_inactive_vehicles()

    def plateau_default_route_uses_current_speeds(self):
        return os.environ.get(PLATEAU_DEFAULT_ROUTE_SPEEDS_ENV, "max").strip().lower() in {
            "current",
            "current_speed",
            "current_speeds",
        }

    def update_map_speeds(self, updated_speeds: dict, last_map_update: timedelta,
                          alternatives_providers: List[AlternativesProvider]) -> Tuple[timedelta, dict]:
        # Get segments where max speeds changed
        # The return value is exactly the segments whose speed_kph was rewritten,
        # which is the only thing that can change a closure.
        temporary_changed = self.sim.routing_map.update_temporary_max_speeds(
            self.sim.setting.departure_time + self.current_offset)
        updated_speeds.update(temporary_changed)
        self.push_store_closures(list(temporary_changed.keys()))

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
    """
    Recomputes travel times for vehicles if map has changed.
    Note: Travel time when finishing a segment is changed in update_followup_route
    """
    current_map_id = routing_map.map_id
    vehicles_to_update = [v_r[0] for v_r in vehicles_with_routes if v_r[0].map_id != current_map_id
                          and v_r[0].alternatives not in [VehicleAlternatives.DEFAULT,
                                                          VehicleAlternatives.DIJKSTRA_SHORTEST]]
    if not vehicles_to_update:
        return

    for provider in alternatives_providers:
        filtered_vehicles = [v for v in vehicles_to_update if v.alternatives == provider.vehicle_behaviour]
        current_routes = [v.get_followup_route() for v in filtered_vehicles]

        travel_times = provider.get_routes_travel_times(current_routes)

        for vehicle, travel_time in zip(filtered_vehicles, travel_times):
            if travel_time is not None:
                vehicle.set_current_travel_time(travel_time, current_map_id)
