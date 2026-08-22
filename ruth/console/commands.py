"""Steering command definitions, validation, and result records.

Syntactic validation (types, ranges, whitelists) runs in the HTTP thread and
rejects bad requests immediately with 400. Semantic validation that needs
simulation state (does the edge exist?) happens later, in the end-of-step
callback, and is reported through the command-result log.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# name -> (validator, description). Validators raise ValueError on bad input
# and return the canonical value to apply.
def _int_range(lo, hi=None):
    def check(v):
        v = int(v)
        if v < lo or (hi is not None and v > hi):
            raise ValueError(f"must be an integer >= {lo}" + (f" and <= {hi}" if hi is not None else ""))
        return v
    return check


def _float_min(lo):
    def check(v):
        v = float(v)
        if v < lo:
            raise ValueError(f"must be a number >= {lo}")
        return v
    return check


def _pos_number(v):
    v = float(v)
    if v <= 0:
        raise ValueError("must be a number > 0")
    return v


PARAM_WHITELIST = {
    "k-alternatives": _int_range(1, 10),
    "map-update-freq-s": _pos_number,
    "los-vehicles-tolerance-s": _float_min(0),
    "travel-time-limit-perc": _float_min(0),
    "stuck-detection": _int_range(0),
}


@dataclass
class Command:
    """A steering command queued by the HTTP thread for the sim callback."""
    id: int
    type: str                    # set-param | restriction | pause | resume | checkpoint | stop
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandResult:
    id: int
    type: str
    status: str                  # pending | ok | error
    detail: str = ""

    def as_dict(self):
        return {"id": self.id, "type": self.type, "status": self.status, "detail": self.detail}


def validate_param(body: dict):
    """Validate a POST /api/param body. Returns (name, canonical_value)."""
    name = body.get("name")
    if name not in PARAM_WHITELIST:
        raise ValueError(f"unknown parameter '{name}'; allowed: {sorted(PARAM_WHITELIST)}")
    if "value" not in body:
        raise ValueError("missing 'value'")
    return name, PARAM_WHITELIST[name](body["value"])


def validate_restriction(body: dict):
    """Validate a POST /api/restriction body. Returns a canonical payload dict."""
    try:
        node_from = int(body["node_from"])
        node_to = int(body["node_to"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("node_from and node_to must be integer OSM node ids")
    speed = float(body.get("speed_kph", -1))
    if speed < 0:
        raise ValueError("speed_kph must be a number >= 0 (0 closes the road)")
    duration = float(body.get("duration_s", 0))
    if duration <= 0:
        raise ValueError("duration_s must be a number > 0")
    start_offset = float(body.get("start_offset_s", 0))
    if start_offset < 0:
        raise ValueError("start_offset_s must be a number >= 0")
    return {
        "node_from": node_from,
        "node_to": node_to,
        "speed_kph": speed,
        "duration_s": duration,
        "start_offset_s": start_offset,
    }


def validate_restriction_delete(body: dict):
    """Validate a POST /api/restriction-delete body."""
    try:
        return {"node_from": int(body["node_from"]), "node_to": int(body["node_to"])}
    except (KeyError, TypeError, ValueError):
        raise ValueError("node_from and node_to must be integer OSM node ids")
