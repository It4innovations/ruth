import os


HETEROGENEOUS_VEHICLES_ENV = "RUTH_ENABLE_HETEROGENEOUS_VEHICLES"
VTC_MOVEMENT_ENV = "RUTH_ENABLE_VTC_MOVEMENT"


def env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def heterogeneous_vehicles_enabled() -> bool:
    return env_flag_enabled(HETEROGENEOUS_VEHICLES_ENV)


def vtc_movement_enabled() -> bool:
    return env_flag_enabled(VTC_MOVEMENT_ENV)
