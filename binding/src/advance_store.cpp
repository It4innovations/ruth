#include "advance_store.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "comm/comm_interface.hpp"

namespace ruthlib {

namespace {

// Kill the JOB, not just this rank.
//
// std::abort() ends one process and leaves every other rank blocked in
// drain_until_quiescent(), waiting on a collective that will never complete --
// so the job hangs until walltime rather than failing. CommInterface::abort()
// is overridden by MPIComm to MPI_Abort, which tears the whole job down; it is
// what ACE's own scheduler calls when a workload throws (scheduler.cpp).
//
// No #ifdef: this file is only ever compiled into ruth and ruthlib, which both
// link ace.
//
// instance() is null until CommInterface::init() runs, and fatal() can fire
// before that (a bad argument during setup_ace) or after finalize(). The
// fallback keeps those cases terminating with the message already printed
// instead of dereferencing null and losing it. comm->abort() is [[noreturn]],
// so the fallback is reached only when ACE is genuinely not up.
[[noreturn]] void fatal() {
  std::fflush(stderr);
  if (auto comm = ace::CommInterface::instance()) {
    comm->abort();
  }
  std::abort();
}

}  // namespace

VehicleStore& VehicleStore::instance() {
  static VehicleStore inst;
  return inst;
}

void VehicleStore::clear() {
  vehicle_id_.clear();
  start_index_.clear();
  start_offset_m_.clear();
  time_offset_us_.clear();
  frequency_us_.clear();
  fcd_period_us_.clear();
  active_.clear();
  dest_node_.clear();
  route_offset_.clear();
  route_len_.clear();
  route_nodes_.clear();
  route_segment_ids_.clear();
  segments_.clear();
  slot_by_vehicle_id_.clear();
}

void VehicleStore::reserve(size_t n_vehicles, size_t n_route_nodes) {
  vehicle_id_.reserve(n_vehicles);
  start_index_.reserve(n_vehicles);
  start_offset_m_.reserve(n_vehicles);
  time_offset_us_.reserve(n_vehicles);
  frequency_us_.reserve(n_vehicles);
  fcd_period_us_.reserve(n_vehicles);
  active_.reserve(n_vehicles);
  dest_node_.reserve(n_vehicles);
  route_offset_.reserve(n_vehicles);
  route_len_.reserve(n_vehicles);
  route_nodes_.reserve(n_route_nodes);
  // One fewer segment than nodes per route, but reserving the same keeps the
  // two buffers growing in step and costs a few bytes per vehicle.
  route_segment_ids_.reserve(n_route_nodes);
  slot_by_vehicle_id_.reserve(n_vehicles);
}

int32_t VehicleStore::add_vehicle(int32_t vehicle_id,
                                  int32_t start_index,
                                  double  start_offset_m,
                                  int64_t time_offset_us,
                                  int64_t frequency_us,
                                  int64_t fcd_period_us,
                                  int32_t active,
                                  int64_t dest_node,
                                  const int64_t* route_nodes,
                                  const int64_t* route_segment_ids,
                                  int32_t route_len) {
  const int32_t slot = static_cast<int32_t>(vehicle_id_.size());

  vehicle_id_.push_back(vehicle_id);
  start_index_.push_back(start_index);
  start_offset_m_.push_back(start_offset_m);
  time_offset_us_.push_back(time_offset_us);
  frequency_us_.push_back(frequency_us);
  fcd_period_us_.push_back(fcd_period_us);
  active_.push_back(active);
  dest_node_.push_back(dest_node);
  route_offset_.push_back(0);
  route_len_.push_back(0);

  slot_by_vehicle_id_.emplace(vehicle_id, slot);

  set_route(slot, start_index, route_nodes, route_segment_ids, route_len);
  return slot;
}

void VehicleStore::set_route(int32_t slot,
                             int32_t start_index,
                             const int64_t* route_nodes,
                             const int64_t* route_segment_ids,
                             int32_t route_len) {
  // Append at the end and repoint; the previous span is left as garbage (see
  // the header note on why this beats compacting on every rewrite).
  route_offset_[slot] = static_cast<int32_t>(route_nodes_.size());
  route_len_[slot] = route_len;
  start_index_[slot] = start_index;

  route_nodes_.insert(route_nodes_.end(), route_nodes, route_nodes + route_len);

  // Keep the two CSR buffers index-aligned: segment i of a route lives at the
  // same offset as node i, so each route contributes route_len entries to both
  // -- its route_len - 1 real segment ids plus one padding slot that is never
  // read (there is no segment leaving the destination).
  //
  // The guard is on route_len > 0, NOT > 1. An EMPTY route must contribute
  // nothing to either buffer. Vehicles with osm_route == [] are real:
  // change_baseline_alternatives empties the route of any vehicle it cannot
  // find a plateau route for, and they stay in sim.vehicles because pruning
  // only happens on the streaming path. Padding those pushed one entry into
  // route_segment_ids_ while route_nodes_ stood still, so every vehicle stored
  // after one of them read a different vehicle's segment ids -- with the right
  // nodes, which is what made it look like a stale route rather than a drift.
  if (route_len > 0) {
    route_segment_ids_.insert(route_segment_ids_.end(),
                              route_segment_ids, route_segment_ids + (route_len - 1));
    route_segment_ids_.push_back(-1);
  }

  // The invariant the whole CSR rests on, checked where it can be broken.
  // O(1), so it stays in release: a drift here is silent and corrupts every
  // later vehicle's task rather than failing near its cause.
  if (route_segment_ids_.size() != route_nodes_.size()) {
    std::fprintf(stderr,
                 "[ruth] VehicleStore::set_route: CSR buffers drifted apart "
                 "(%zu segment ids vs %zu nodes) writing vehicle %d with route_len %d.\n",
                 route_segment_ids_.size(), route_nodes_.size(),
                 vehicle_id_[slot], route_len);
    fatal();
  }
}

void VehicleStore::resize_segments(size_t max_routing_id) {
  segments_.resize(max_routing_id + 1);
}

void VehicleStore::set_segment(int64_t routing_id, double length_m, double max_speed_kph,
                               int32_t lanes, int32_t is_closed) {
  if (routing_id < 0 || static_cast<size_t>(routing_id) >= segments_.size()) {
    std::fprintf(stderr,
                 "[ruth] VehicleStore::set_segment: routing_id %lld out of range "
                 "(table holds %zu entries); call resize_segments() first.\n",
                 static_cast<long long>(routing_id), segments_.size());
    fatal();
  }
  SegmentGeom& geom = segments_[static_cast<size_t>(routing_id)];
  geom.length_m = length_m;
  geom.max_speed_kph = max_speed_kph;
  geom.lanes = lanes;
  geom.is_closed = is_closed;
  geom.valid = 1;
}

size_t VehicleStore::update_movement(const int32_t* vehicle_id,
                                     const int32_t* start_index,
                                     const double*  start_offset_m,
                                     const int64_t* time_offset_us,
                                     const int32_t* active,
                                     size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const int32_t slot = slot_of(vehicle_id[i]);
    if (slot < 0) {
      std::fprintf(stderr,
                   "[ruth] VehicleStore::update_movement: vehicle %d is not in the "
                   "store. The store and the Python vehicle list have diverged; "
                   "re-initialise the store after the population changes "
                   "(streaming load or prune).\n",
                   vehicle_id[i]);
      fatal();
    }
    start_index_[slot] = start_index[i];
    start_offset_m_[slot] = start_offset_m[i];
    time_offset_us_[slot] = time_offset_us[i];
    active_[slot] = active[i];
  }
  return n;
}

void VehicleStore::set_segments_closed(const int64_t* routing_id, const int32_t* is_closed,
                                       size_t n) {
  for (size_t i = 0; i < n; ++i) {
    set_segment_closed(routing_id[i], is_closed[i]);
  }
}

void VehicleStore::set_routes(const int32_t* vehicle_id,
                              const int32_t* start_index,
                              const int64_t* route_nodes,
                              const int64_t* route_segment_ids,
                              const int32_t* route_offsets,
                              size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const int32_t slot = slot_of(vehicle_id[i]);
    if (slot < 0) {
      std::fprintf(stderr,
                   "[ruth] VehicleStore::set_routes: vehicle %d is not in the store.\n",
                   vehicle_id[i]);
      fatal();
    }
    const int32_t begin = route_offsets[i];
    const int32_t len = route_offsets[i + 1] - begin;
    set_route(slot, start_index[i], route_nodes + begin, route_segment_ids + begin, len);
  }
}

std::vector<AdvanceTask> VehicleStore::make_tasks(const int32_t* vehicle_id,
                                                  size_t n) const {
  std::vector<AdvanceTask> tasks;
  tasks.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    const int32_t slot = slot_of(vehicle_id[i]);
    if (slot < 0) {
      std::fprintf(stderr,
                   "[ruth] VehicleStore::make_tasks: vehicle %d is not in the store.\n",
                   vehicle_id[i]);
      fatal();
    }
    tasks.push_back(make_task(slot));
  }
  return tasks;
}

VehicleStore::DeltaOutcome VehicleStore::apply_deltas(const int32_t* vehicle_id,
                                                      const int32_t* new_start_index,
                                                      const double*  new_offset_m,
                                                      const int64_t* new_time_offset_us,
                                                      const int32_t* moved,
                                                      const int32_t* changed_segment,
                                                      const int32_t* reached_end,
                                                      size_t n) {
  DeltaOutcome outcome;

  for (size_t i = 0; i < n; ++i) {
    const int32_t slot = slot_of(vehicle_id[i]);
    if (slot < 0) {
      std::fprintf(stderr,
                   "[ruth] VehicleStore::apply_deltas: vehicle %d is not in the store.\n",
                   vehicle_id[i]);
      fatal();
    }

    // Read the OLD index before overwriting it: the queue the vehicle leaves is
    // keyed by the segment it was on, not the one it is on now.
    const int32_t old_index = start_index_[slot];
    const int32_t base = route_offset_[slot];

    start_index_[slot] = new_start_index[i];
    start_offset_m_[slot] = new_offset_m[i];
    time_offset_us_[slot] = new_time_offset_us[i];

    if (moved[i] != 0) {
      ++outcome.moved_count;
    }

    if (changed_segment[i] != 0) {
      // route.py indexes osm_route[old_index] and [old_index + 1] unguarded; a
      // changed_segment delta implies both exist, so an out-of-range pair here
      // means the store's route disagrees with the one the task was built from.
      if (old_index + 1 >= route_len_[slot]) {
        std::fprintf(stderr,
                     "[ruth] VehicleStore::apply_deltas: vehicle %d changed segment from "
                     "index %d but its route holds only %d nodes.\n",
                     vehicle_id[i], old_index, route_len_[slot]);
        fatal();
      }
      outcome.removals.push_back(QueueEvent{vehicle_id[i],
                                            route_nodes_[base + old_index],
                                            route_nodes_[base + old_index + 1]});
    }

    if (reached_end[i] != 0 && moved[i] != 0) {
      // Decided from the NEW position: next_node is what the vehicle is heading
      // to now, and reaching the destination ends its run.
      const int64_t next = next_node(slot);
      if (next == dest_node(slot)) {
        active_[slot] = 0;
      } else {
        outcome.additions.push_back(QueueEvent{vehicle_id[i],
                                               current_node(slot), next});
      }
    }
  }
  return outcome;
}

void VehicleStore::advance_waiting(const int32_t* vehicle_id, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const int32_t slot = slot_of(vehicle_id[i]);
    if (slot < 0) {
      std::fprintf(stderr,
                   "[ruth] VehicleStore::advance_waiting: vehicle %d is not in the store.\n",
                   vehicle_id[i]);
      fatal();
    }
    time_offset_us_[slot] += frequency_us_[slot];
  }
}

void VehicleStore::route_view(int32_t slot,
                              std::vector<int64_t>& out_nodes,
                              std::vector<int64_t>& out_segment_ids) const {
  const int32_t base = route_offset_[slot];
  const int32_t len = route_len_[slot];

  out_nodes.assign(route_nodes_.begin() + base, route_nodes_.begin() + base + len);
  // len - 1 real segment ids; the trailing slot is the alignment padding.
  const int32_t n_segments = len > 0 ? len - 1 : 0;
  out_segment_ids.assign(route_segment_ids_.begin() + base,
                         route_segment_ids_.begin() + base + n_segments);
}

void VehicleStore::set_segment_closed(int64_t routing_id, int32_t is_closed) {
  if (routing_id < 0 || static_cast<size_t>(routing_id) >= segments_.size()) {
    return;  // a closure on a segment no vehicle routes over is not an error
  }
  segments_[static_cast<size_t>(routing_id)].is_closed = is_closed;
}

const SegmentGeom& VehicleStore::segment(int64_t routing_id) const {
  if (routing_id < 0 || static_cast<size_t>(routing_id) >= segments_.size() ||
      segments_[static_cast<size_t>(routing_id)].valid == 0) {
    std::fprintf(stderr,
                 "[ruth] VehicleStore::segment: missing geometry for routing_id %lld "
                 "(table holds %zu entries).\n",
                 static_cast<long long>(routing_id), segments_.size());
    fatal();
  }
  return segments_[static_cast<size_t>(routing_id)];
}

int32_t VehicleStore::slot_of(int32_t vehicle_id) const {
  const auto it = slot_by_vehicle_id_.find(vehicle_id);
  return it == slot_by_vehicle_id_.end() ? -1 : it->second;
}

void VehicleStore::set_position(int32_t slot, int32_t start_index, double start_offset_m) {
  start_index_[slot] = start_index;
  start_offset_m_[slot] = start_offset_m;
}

void VehicleStore::set_time_offset_us(int32_t slot, int64_t time_offset_us) {
  time_offset_us_[slot] = time_offset_us;
}

void VehicleStore::set_active(int32_t slot, int32_t active) {
  active_[slot] = active;
}

int64_t VehicleStore::current_node(int32_t slot) const {
  return route_nodes_[route_offset_[slot] + start_index_[slot]];
}

int64_t VehicleStore::next_node(int32_t slot) const {
  const int32_t index = start_index_[slot] + 1;
  if (index >= route_len_[slot]) {
    return -1;  // Vehicle.next_node returns None here
  }
  return route_nodes_[route_offset_[slot] + index];
}

int64_t VehicleStore::dest_node(int32_t slot) const {
  return dest_node_[slot];
}

bool VehicleStore::at_segment_end(int32_t slot) const {
  const int32_t n_segments = route_len_[slot] - 1;
  if (start_index_[slot] >= n_segments) {
    return false;  // no current segment left to be at the end of
  }
  const int64_t sid = route_segment_ids_[route_offset_[slot] + start_index_[slot]];
  // Exact comparison on purpose: route.py's advance_vehicle snaps the position
  // to the segment length when it finishes one, so equality is how "finished"
  // is represented rather than an approximation of it.
  return start_offset_m_[slot] == segment(sid).length_m;
}

void VehicleStore::fill_segment_desc(SegmentDesc& desc, int64_t routing_id) const {
  const SegmentGeom& geom = segment(routing_id);
  desc.segment_id = routing_id;
  desc.length_m = geom.length_m;
  desc.max_speed_kph = geom.max_speed_kph;
  desc.lanes = geom.lanes;
  desc.valid = 1;
  desc.is_closed = geom.is_closed;
  desc._pad = 0;
}

namespace {

// Python's round() is round-half-to-EVEN (round(2.5) == 2, round(3.5) == 4),
// not the round-half-away-from-zero that std::round gives. Selection compares
// the rounded offset for exact equality, so getting this wrong would silently
// drop or add vehicles on any step whose offset lands exactly on a half
// frequency -- which, with whole-second frequencies and offsets, is not a rare
// corner but a regular occurrence.
//
// Written out rather than leaning on std::nearbyint + FE_TONEAREST: that would
// be correct only while nothing else in the process has changed the floating
// point rounding mode, and a dependency on ambient global state is exactly what
// this should not have.
int64_t round_half_to_even(double x) {
  const double floor_x = std::floor(x);
  const double frac = x - floor_x;
  const int64_t below = static_cast<int64_t>(floor_x);

  if (frac > 0.5) return below + 1;
  if (frac < 0.5) return below;
  return (below % 2 == 0) ? below : below + 1;   // exactly .5 -> the even side
}

// timedelta.total_seconds() is ((days*86400 + seconds)*10**6 + microseconds) /
// 10**6: exact integer arithmetic followed by a single division. Converting the
// microsecond count to double is exact below 2^53 us (~285 years of simulated
// time) and 1e6 is exactly representable, so this reproduces the Python float
// bit for bit rather than merely closely.
double total_seconds(int64_t time_offset_us) {
  return static_cast<double>(time_offset_us) / 1e6;
}

}  // namespace

std::vector<int32_t> VehicleStore::select_within_offset(int64_t offset_seconds,
                                                        int64_t freq_seconds) const {
  std::vector<int32_t> selected;
  if (freq_seconds == 0) {
    return selected;  // would divide by zero; Python would raise
  }

  for (size_t slot = 0; slot < vehicle_id_.size(); ++slot) {
    if (active_[slot] == 0) {
      continue;
    }
    const double scaled = total_seconds(time_offset_us_[slot]) /
                          static_cast<double>(freq_seconds);
    const int64_t rounded = freq_seconds * round_half_to_even(scaled);
    if (rounded == offset_seconds) {
      selected.push_back(vehicle_id_[slot]);
    }
  }
  return selected;
}

bool VehicleStore::min_active_time_offset_us(int64_t& out_time_offset_us) const {
  // compute_current_offset() is
  //     min(filter(None, map(lambda v: v.time_offset if v.active else None, ...)))
  // and filter(None, ...) drops every FALSY value -- which includes
  // timedelta(0), not just the None it looks like it is filtering. So a vehicle
  // sitting at exactly offset zero does not participate. That is reproduced
  // here deliberately: this is a port, and changing the rule would change which
  // step the simulation runs next.
  bool found = false;
  int64_t best = 0;

  for (size_t slot = 0; slot < vehicle_id_.size(); ++slot) {
    if (active_[slot] == 0 || time_offset_us_[slot] == 0) {
      continue;
    }
    if (!found || time_offset_us_[slot] < best) {
      best = time_offset_us_[slot];
      found = true;
    }
  }

  if (found) {
    out_time_offset_us = best;
  }
  return found;
}

int32_t VehicleStore::count_active() const {
  int32_t n = 0;
  for (const int32_t a : active_) {
    n += (a != 0) ? 1 : 0;
  }
  return n;
}

std::vector<int32_t> VehicleStore::at_segment_end_ids(
    const std::vector<int32_t>& vehicle_ids) const {
  std::vector<int32_t> out;
  out.reserve(vehicle_ids.size());

  for (const int32_t id : vehicle_ids) {
    const int32_t slot = slot_of(id);
    if (slot >= 0 && at_segment_end(slot)) {
      out.push_back(id);
    }
  }
  return out;
}

AdvanceTask VehicleStore::make_task(int32_t slot) const {
  AdvanceTask task{};

  task.vehicle_id = vehicle_id_[slot];
  task.start_index = start_index_[slot];
  task.start_offset_m = start_offset_m_[slot];
  task.time_offset_us = time_offset_us_[slot];
  task.frequency_us = frequency_us_[slot];
  task.fcd_period_us = fcd_period_us_[slot];

  // _fill_task takes osm_route[idx:idx+3], i.e. up to two segments: the current
  // one and the next. Fewer than two remaining leaves seg1 zeroed with
  // valid == 0, which the kernel reads as "next segment missing" (closed).
  const int32_t base = route_offset_[slot];
  const int32_t index = start_index_[slot];
  const int32_t n_segments = route_len_[slot] - 1;

  if (index < n_segments) {
    fill_segment_desc(task.seg0, route_segment_ids_[base + index]);
  }
  if (index + 1 < n_segments) {
    fill_segment_desc(task.seg1, route_segment_ids_[base + index + 1]);
  }
  return task;
}

}  // namespace ruthlib
