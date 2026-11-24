#pragma once

// POD payloads exchanged between master and workers for the distributed
// vehicle-advance model. All are flat and trivially memcpy-able so they can be
// marshalled with the helpers in advance_marshal.hpp and gathered raw via
// CommInterface::sync_allgatherv.

#include <cstdint>
#include <type_traits>

#include "globalview.hpp"   // FCDRecord

namespace ruthlib {

/// One segment of a vehicle's local route window, resolved by the master.
/// The worker needs no access to the routing graph: everything the advance
/// kernel reads about geometry travels in the task.
struct SegmentDesc {
  int64_t segment_id;        ///< GlobalView key (Map.segment_id_to_int)
  double  length_m;          ///< Segment.length
  double  max_speed_kph;     ///< Segment.max_allowed_speed_kph (ORIGINAL limit,
                             ///< used for the moving speed = max_speed_kph * LoS)
  int32_t lanes;             ///< Segment.lanes
  int32_t valid;             ///< 1 if this slot holds a real segment, else 0
  int32_t is_closed;         ///< 1 if the CURRENT max speed is 0 (get_current_max_speed);
                             ///< used for the has_next_segment_closed check
  int32_t _pad;              ///< explicit trailing padding: the 8-byte alignment
                             ///< would add this anyway; naming it makes the 40-byte
                             ///< size explicit so the numpy dtype layout is obvious
};

/// Everything needed to advance one vehicle by one step. `time_offset_us`,
/// `start_index`, `start_offset_m` are the vehicle's *current* state; the
/// worker returns the updated values in an AdvanceDelta.
struct AdvanceTask {
  int32_t     vehicle_id;
  int32_t     start_index;       ///< index into the vehicle's osm_route
  double      start_offset_m;    ///< start_distance_offset (position on segment)
  // Time is carried as integer MICROSECONDS to match Python timedelta exactly.
  // Using double seconds accumulates sub-second drift that the LoS tolerance
  // window then amplifies into divergent trajectories.
  int64_t     time_offset_us;    ///< current time_offset, in microseconds
  int64_t     frequency_us;      ///< vehicle.frequency, in microseconds
  int64_t     fcd_period_us;     ///< vehicle.fcd_sampling_period, in microseconds
  SegmentDesc seg0;              ///< current segment (driving_route_part[0])
  SegmentDesc seg1;              ///< next segment    (driving_route_part[1])
};

/// Result of advancing one vehicle: the new movement state. Queue bookkeeping
/// and history are derived from this on the master.
struct AdvanceDelta {
  int32_t vehicle_id;
  int32_t new_start_index;
  double  new_offset_m;          ///< new start_distance_offset
  int64_t new_time_offset_us;    ///< new time_offset, in microseconds
  int32_t moved;                 ///< 1 if position changed this step
  int32_t changed_segment;       ///< 1 if it advanced onto seg1
  int32_t reached_end;           ///< 1 if it finished its current segment this step
  int32_t _pad;
  // NOTE: arrival (vehicle.active) is decided in Python during replay, which
  // owns osm_route/dest_node; the kernel reports pure kinematics only.
};

static_assert(std::is_trivially_copyable<SegmentDesc>::value,  "SegmentDesc must be POD");
static_assert(std::is_trivially_copyable<AdvanceTask>::value,  "AdvanceTask must be POD");
static_assert(std::is_trivially_copyable<AdvanceDelta>::value, "AdvanceDelta must be POD");
static_assert(std::is_trivially_copyable<FCDRecord>::value,    "FCDRecord must be POD");

}  // namespace ruthlib
