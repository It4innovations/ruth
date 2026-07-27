#include "advance_kernel.hpp"

#include <cmath>
#include <cstdint>

namespace ruthlib {

// segment_end_delta in route.py:move_on_segment defaults to 2 metres.
static constexpr double SEGMENT_END_DELTA_M = 2.0;

double speed_kph_to_mps(double kph) { return kph * (1000.0 / 3600.0); }

namespace {

// FCD/LoS timestamps are epoch SECONDS as a double (the GlobalView stores them
// that way). We carry time internally as integer microseconds to match Python
// timedelta exactly, and convert to seconds only at the GV boundary with a
// single division of the total microsecond count — identical to what
// datetime_to_timestamp() does in the serial path, so the doubles match.
inline double epoch_us_to_seconds(int64_t us) { return static_cast<double>(us) / 1e6; }

// Build the "did not move this step" delta: keep position, advance one
// frequency. Shared by every early-out branch in route.py that just shifts the
// clock (next segment closed, stuck in a jam, zero speed, first-move guard).
//
// route.py still runs a stalled vehicle through generate_fcds (speed 0), so it
// emits stationary FCDs on its current segment: one final sample, plus interior
// samples per fcd_period while the position is short of the segment end. The
// sole exception is route.py's first-move guard — a vehicle still at the very
// start of its route (index 0, offset 0) produces no FCDs. Those stationary
// FCDs matter: a stopped car must keep occupying its segment in the GlobalView,
// or LoS would wrongly show the jam clearing.
AdvanceDelta stalled(const AdvanceTask& t, int64_t current_epoch_us,
                     std::vector<FCDRecord>& out_fcds) {
  const bool first_move_guard = (t.start_index == 0 && t.start_offset_m == 0.0);
  if (!first_move_guard) {
    const int64_t end_epoch_us = current_epoch_us + t.frequency_us;
    const double pos = t.start_offset_m;   // stationary: step_m == 0, pos is constant
    int64_t ts_us = current_epoch_us;
    // interior samples (route.py generate_fcds while-loop with step_m == 0)
    while ((ts_us + t.fcd_period_us) < end_epoch_us && pos < t.seg0.length_m) {
      ts_us += t.fcd_period_us;
      out_fcds.emplace_back(epoch_us_to_seconds(ts_us), t.vehicle_id, t.seg0.segment_id, pos, 0.0);
    }
    // final end-of-window sample
    out_fcds.emplace_back(epoch_us_to_seconds(end_epoch_us), t.vehicle_id, t.seg0.segment_id, pos, 0.0);
  }

  AdvanceDelta d{};
  d.vehicle_id = t.vehicle_id;
  d.new_start_index = t.start_index;
  d.new_offset_m = t.start_offset_m;
  d.new_time_offset_us = t.time_offset_us + t.frequency_us;
  d.moved = 0;
  d.changed_segment = 0;
  d.reached_end = 0;
  return d;
}

}  // namespace

AdvanceDelta advance_one_vehicle(const AdvanceTask& t,
                                 const GlobalView& gv,
                                 int64_t departure_epoch_us,
                                 double los_tolerance_s,
                                 std::vector<FCDRecord>& out_fcds) {
  const int64_t current_epoch_us = departure_epoch_us + t.time_offset_us;
  const double current_time_s = epoch_us_to_seconds(current_epoch_us);
  // Distance/step use frequency and fcd_period in seconds (double), exactly as
  // route.py (freq*speed, step_m = speed*period). For whole-second frequencies
  // this equals route.py's int _frequency_seconds.
  const double frequency_s = static_cast<double>(t.frequency_us) / 1e6;
  const double fcd_period_s = static_cast<double>(t.fcd_period_us) / 1e6;

  // ---- get_vehicle_speed -------------------------------------------------
  const SegmentDesc* cur = &t.seg0;
  double start_position = t.start_offset_m;
  bool changed_segment = false;

  if (start_position >= cur->length_m) {  // at the end of seg0
    // has_next_segment_closed: next segment missing, or its CURRENT max speed is
    // 0 (a closure lives in the current network, not the original limit).
    const bool next_closed = (t.seg1.valid == 0) || (t.seg1.is_closed != 0);
    if (next_closed) {
      return stalled(t, current_epoch_us, out_fcds);
    }
    start_position = 0.0;
    changed_segment = true;
    cur = &t.seg1;
  }

  const double los = gv.level_of_service_in_front_of_vehicle(
      current_time_s, cur->segment_id, cur->length_m, cur->lanes,
      los_tolerance_s, t.vehicle_id, start_position);
  if (std::isinf(los)) {
    return stalled(t, current_epoch_us, out_fcds);  // stuck in a jam
  }
  const double speed_mps = speed_kph_to_mps(cur->max_speed_kph * los);
  if (speed_mps == 0.0) {
    return stalled(t, current_epoch_us, out_fcds);
  }

  // ---- move_on_segment ---------------------------------------------------
  const int new_index = changed_segment ? (t.start_index + 1) : t.start_index;
  const double seg_len = cur->length_m;
  const double raw_end = start_position + frequency_s * speed_mps;

  int64_t end_epoch_us;
  double end_position;
  if (raw_end >= (seg_len - SEGMENT_END_DELTA_M)) {
    // finishes the segment (snap to the exact end). travel_time is rounded to
    // whole microseconds, matching route.py's timedelta(seconds=travel_time).
    const double travel_distance_m = seg_len - start_position;
    const double travel_time_s = travel_distance_m / speed_mps;
    end_epoch_us = current_epoch_us + static_cast<int64_t>(std::llround(travel_time_s * 1e6));
    end_position = seg_len;
  } else {
    end_epoch_us = current_epoch_us + t.frequency_us;
    end_position = raw_end;
  }

  // route.py advance_vehicle "first move" guard: index 0 & pos 0 => no move
  if (new_index == 0 && end_position == 0.0) {
    return stalled(t, current_epoch_us, out_fcds);
  }

  // The segment is finished exactly when the kernel snapped to its end.
  const bool reached_segment_end = (end_position == seg_len);

  // ---- generate_fcds -----------------------------------------------------
  // Sampled FCDs along the movement on `cur`, then the end-of-move point.
  const double step_m = speed_mps * fcd_period_s;
  double pos = start_position;
  int64_t ts_us = current_epoch_us;
  while ((ts_us + t.fcd_period_us) < end_epoch_us && (pos + step_m) < seg_len) {
    pos += step_m;
    ts_us += t.fcd_period_us;
    out_fcds.emplace_back(epoch_us_to_seconds(ts_us), t.vehicle_id, cur->segment_id, pos, speed_mps);
  }
  out_fcds.emplace_back(epoch_us_to_seconds(end_epoch_us), t.vehicle_id, cur->segment_id, end_position, speed_mps);

  AdvanceDelta d{};
  d.vehicle_id = t.vehicle_id;
  d.new_start_index = new_index;
  d.new_offset_m = end_position;
  d.new_time_offset_us = t.time_offset_us + (end_epoch_us - current_epoch_us);
  d.moved = 1;
  d.changed_segment = changed_segment ? 1 : 0;
  d.reached_end = reached_segment_end ? 1 : 0;
  return d;
}

}  // namespace ruthlib
