
from enum import Enum
from datetime import timedelta
from functools import partial
from dataclasses import dataclass
from typing import Any, Callable, List, Dict, Optional, NewType

from probduration import VehiclePlan, avg_delays

from .common import distance_duration
from ..vehicle import Vehicle


Comparable = NewType("Comparable", Any)


@dataclass
class RouteRankingAlgorithm:
    """Route ranking algorithm with optional data preparation function.

    Parameters:
    -----------
        rank_route: Callable[[List, Dict], Comparable]
          A function that rank a route. The positional and keyword arguments are "algorithm specific"
        prepare_data: Optional[Callable[[List, Dict], Any]]
          A function that can transform input data hence perform a preprocessing for ranking function. By default
          it is assigned identity function.
    """
    rank_route: Callable[[List, Dict], Comparable]
    prepare_data: Optional[Callable[[List, Dict], Any]] = lambda data: data


def duration_based_on_global_view(gv_db, vehicle_plan):
    """Rank the route by passing a vehicle over it and compute the duration based on global view.

    Parameters:
    -----------
        gv_db: GlobalViewDb
        vehicle_plan: Tuple[Vehicle, VehiclePlan]
    """
    _, plan = vehicle_plan
    # NOTE: empty random generator is valid as the global view LoS database does not use it
    duration, *_ = distance_duration(plan.route, plan.departure_time, gv_db, lambda: None)

    if duration == float('inf'):
        return timedelta.max - timedelta(days=7)

    return duration


def adjust_plan_by_global_view(vehicle_plan: (Vehicle, VehiclePlan), distance, ff_db, gv_db, rnd_gen):
    vehicle, plan = vehicle_plan
    ff_dur, *_ = distance_duration(plan.route, plan.departure_time, ff_db, rnd_gen, distance)
    duration, position, los = distance_duration(plan.route, plan.departure_time, gv_db, rnd_gen, distance)

    if duration == float('inf'):
        delay = timedelta.max - timedelta(days=7)  # NOTE: decrease the delay in order to be able to add to the number;
                                                   #       assumption: there is no forcasted delay longer than 7 days.
        duration = vehicle.frequency
        position = plan.start_position
        los = 0.0
    else:
        delay = duration - ff_dur

    return vehicle, VehiclePlan(plan.id, plan.route, position, plan.departure_time + duration), los, delay


def precompute_prob_delays(vehicle_plans, gv_db, gv_distance, ff_db, pp_db, n_samples, rnd_gen):
    """ Precompute the information for `probable_delay` route-ranking function. The function returns the vehicle plans
    extended by global view LoS and delay, plus the probable delay on rest of the route.

    Parameters:
    -----------
        vehicle_plans: List[Tuple[Vehicle, VehiclePlan]]
          A list of vehicle plans with paired vehicle. It is separated as the VehiclePlan is bound type from Rust
          (not all the information in vehicle must be passed into the Rust library).
        gv_db: GlobalViewDb
        gv_distance: float
          A distance a vehicle is moved based on global view restrictions
        ff_db: FreeFlowDb
        pp_db: ProbProfileDb
        n_samples: int
          A number of samples of Monte Carlo Simulation
        rnd_gen: Callable[[], float]
          A random generator which returns values from 0.0..1.0 interval
    """
    # move the vehicle by specified distance and adjust its plan
    vehicles, plans, loses, gv_delays = zip(*map(partial(adjust_plan_by_global_view,
                                                         distance=gv_distance,
                                                         gv_db=gv_db,
                                                         ff_db=ff_db,
                                                         rnd_gen=rnd_gen),
                                                 vehicle_plans))

    prob_delays = avg_delays(list(plans), pp_db.prob_profiles, n_samples)

    return list(zip(vehicles, plans, loses, gv_delays, prob_delays))


def probable_delay(extended_vehicle_plan):
    """Rank the route based global view delay compbined with the probable delay.

    Parameteres:
    ------------
        extended_vehicle_plan: Tuple[Vehicle, VehiclePlan, float, float, float]
          A vehicle plan extended with global view LoS and delay, plus probable delay.
    """
    _, _, gv_los, gv_delay, prob_delay = extended_vehicle_plan
    if gv_los < 1.0:
        # NOTE: the more the gv_los is closer to zero the more important is the global view delay
        #       as the vehicle hit traffic jam or closed road
        return gv_delay + prob_delay / (1.0 - gv_los)
    return gv_delay + prob_delay


class RouteRankingAlgorithms(Enum):
    """An enumeration of available route ranking algorithm. Each algorithm can be accompanied by a data/preprocessing
     function; by default the function is identity.

    Attributes:
    -----------
        DURATION: RouteRankingAlgorithm
          Compute duration on a route at a departure time based on global view.
        PROBABLE_DELAY: RouteRankingAlgorithm
          Compute probable delay on a route at a departure time using combination of global view for first X meters and
          probable delay on the rest of the route. The algorithm takes advantage of Monte Carlo Simulation.
    """
    DURATION = RouteRankingAlgorithm(duration_based_on_global_view)
    PROBABLE_DELAY = RouteRankingAlgorithm(probable_delay, precompute_prob_delays)
