#include "advance_store.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ruthlib {

const SegmentGeom VehicleStore::kMissingSegment{};

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
  if (route_len > 1) {
    route_segment_ids_.insert(route_segment_ids_.end(),
                              route_segment_ids, route_segment_ids + (route_len - 1));
  }
  // Keep the two CSR buffers index-aligned: segment i of a route lives at the
  // same offset as node i, so the last node's slot is padding that is never
  // read (there is no segment leaving the destination).
  route_segment_ids_.push_back(-1);
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
    std::fflush(stderr);
    std::abort();
  }
  SegmentGeom& geom = segments_[static_cast<size_t>(routing_id)];
  geom.length_m = length_m;
  geom.max_speed_kph = max_speed_kph;
  geom.lanes = lanes;
  geom.is_closed = is_closed;
  geom.valid = 1;
}

void VehicleStore::set_segment_closed(int64_t routing_id, int32_t is_closed) {
  if (routing_id < 0 || static_cast<size_t>(routing_id) >= segments_.size()) {
    return;  // a closure on a segment no vehicle routes over is not an error
  }
  segments_[static_cast<size_t>(routing_id)].is_closed = is_closed;
}

const SegmentGeom& VehicleStore::segment(int64_t routing_id) const {
  if (routing_id < 0 || static_cast<size_t>(routing_id) >= segments_.size()) {
    return kMissingSegment;
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
  return route_nodes_[route_offset_[slot] + route_len_[slot] - 1];
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
