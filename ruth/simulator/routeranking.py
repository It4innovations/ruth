from datetime import timedelta
from functools import partial

from probduration import VehiclePlan, avg_delays

from ..distsim import distance_duration
from ..vehicle import Vehicle


def duration_based_on_global_view(gv_db, vehicle_plan):
    """Rank the route by passing a vehicle over it and compute the duration based on global view.

    Parameters:
    -----------
        gv_db: GlobalViewDb
        vehicle_plan: Tuple[Vehicle, VehiclePlan]
    """
    _, plan = vehicle_plan
    duration, *_ = distance_duration(plan.route, plan.departure_time, gv_db)

    if duration == float("inf"):
        return timedelta.max

    return duration


def adjust_plan_by_global_view(vehicle_plan: (Vehicle, VehiclePlan), distance, ff_db, gv_db):
    vehicle, plan = vehicle_plan
    ff_dur, *_ = distance_duration(plan.route, plan.departure_time, ff_db, distance)
    duration, position, los = distance_duration(plan.route, plan.departure_time, gv_db, distance)

    if duration == float("inf"):
        delay = timedelta.max
        duration = vehicle.frequency
        position = plan.start_position
        los = 0.0
    else:
        delay = duration - ff_dur

    return vehicle, VehiclePlan(plan.id, plan.route, position, plan.departure_time + duration), los, delay


def precompute_prob_delays(vehicle_plans, gv_db, gv_distance, ff_db, pp_db, n_samples):
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
    """
    # move the vehicle by specified distance and adjust its plan
    vehicles, plans, loses, gv_delays = zip(*map(partial(adjust_plan_by_global_view,
                                                         distance=gv_distance, gv_db=gv_db, ff_db=ff_db),
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
