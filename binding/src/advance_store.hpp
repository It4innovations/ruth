#pragma once

// Columnar, C++-resident vehicle movement state for the distributed advance.
//
// WHY THIS EXISTS. Today the master rebuilds every step's AdvanceTask array from
// Python Vehicle objects (mpi_advance._build_task_array) and writes every delta
// back into them (_apply_delta). That is O(active vehicles) of Python per step
// no matter how many ranks exist, and it is what caps the distributed advance at
// roughly 1.3x however well the workers scale. VehicleStore holds the same state
// in flat arrays on every rank so a work unit can be a SLOT RANGE rather than a
// marshalled payload, and so task construction and delta replay stop crossing
// the Python boundary at all.
//
// NOTHING USES THIS YET. It is introduced on its own, unreferenced, so it can be
// reviewed and unit-tested against the Python it will eventually replace before
// anything depends on it. make_task() is included precisely so that can be
// checked: it must reproduce mpi_advance._fill_task exactly, field for field.
//
// OWNERSHIP RULE (for later patches): a column has exactly one writer, and the
// writer changes only at a barrier. Movement columns (start_index,
// start_offset_m, time_offset_us, active) are written by the advance phase;
// route columns are written by the alternatives phase. Never both at once.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "advance_types.hpp"

namespace ruthlib {

/// Geometry and current closure state of one map segment, keyed by routing_id
/// (Map.segment_id_to_int; dense and 1-based, see Map._load).
struct SegmentGeom {
  double  length_m       = 0.0;
  double  max_speed_kph  = 0.0;  ///< ORIGINAL limit; the moving speed is this * LoS
  int32_t lanes          = 0;
  int32_t is_closed      = 0;    ///< current max speed == 0 (closures live in the
                                 ///< current network, not the original limit)
  int32_t valid          = 0;    ///< 0 until set_segment() fills this slot
};

class VehicleStore {
public:
  static VehicleStore& instance();

  VehicleStore() = default;
  VehicleStore(const VehicleStore&) = delete;
  VehicleStore& operator=(const VehicleStore&) = delete;

  // --- population ----------------------------------------------------------

  void clear();
  void reserve(size_t n_vehicles, size_t n_route_nodes);

  /// Append one vehicle and return its slot.
  ///
  /// `route_nodes` is the vehicle's osm_route (OSM node ids, length
  /// `route_len`). `route_segment_ids` holds the routing_id of each segment of
  /// that route, so it has length `route_len - 1`: segment i spans
  /// route_nodes[i] -> route_nodes[i+1]. Carrying the segment ids explicitly
  /// means make_task() never has to look an edge up by node pair, which is the
  /// networkx dict-of-dict cost that dominates _fill_task today.
  int32_t add_vehicle(int32_t vehicle_id,
                      int32_t start_index,
                      double  start_offset_m,
                      int64_t time_offset_us,
                      int64_t frequency_us,
                      int64_t fcd_period_us,
                      int32_t active,
                      const int64_t* route_nodes,
                      const int64_t* route_segment_ids,
                      int32_t route_len);

  /// Replace a vehicle's route (used after route selection picks a new one).
  ///
  /// The new route is APPENDED to the CSR buffers and the vehicle's span is
  /// repointed; the old span becomes garbage. That keeps a rewrite O(new route
  /// length) instead of memmoving every later vehicle's route, at the cost of
  /// buffer growth -- compaction belongs with the prune patch, not here.
  void set_route(int32_t slot,
                 int32_t start_index,
                 const int64_t* route_nodes,
                 const int64_t* route_segment_ids,
                 int32_t route_len);

  // --- segment table -------------------------------------------------------

  /// Size the table to hold routing ids 0..max_routing_id (id 0 is unused).
  void resize_segments(size_t max_routing_id);
  void set_segment(int64_t routing_id, double length_m, double max_speed_kph,
                   int32_t lanes, int32_t is_closed);
  /// Update just the closure flag, as update_current_speeds does per step.
  void set_segment_closed(int64_t routing_id, int32_t is_closed);

  /// Bulk form of set_segment_closed. Only `speed_kph` in the current network
  /// decides a closure, and only Map.update_temporary_max_speeds writes it, so
  /// the caller passes exactly the segments that call reported as changed.
  void set_segments_closed(const int64_t* routing_id, const int32_t* is_closed, size_t n);

  /// Replace the routes of `n` vehicles addressed by id, from a CSR block.
  /// Used when route selection rewrites a route mid-run.
  void set_routes(const int32_t* vehicle_id,
                  const int32_t* start_index,
                  const int64_t* route_nodes,
                  const int64_t* route_segment_ids,
                  const int32_t* route_offsets,
                  size_t n);

  /// Build tasks for `n` vehicles addressed by id, in the order given.
  std::vector<AdvanceTask> make_tasks(const int32_t* vehicle_id, size_t n) const;

  /// This vehicle's route as the store holds it: the node span and the
  /// routing_id of each of its segments. For diagnosing a task mismatch, where
  /// the question is almost always "is the store's route the current one".
  void route_view(int32_t slot,
                  std::vector<int64_t>& out_nodes,
                  std::vector<int64_t>& out_segment_ids) const;

  /// Total slots ever allocated to route storage, and how many are still
  /// referenced. set_route appends rather than compacting, so a long run with
  /// frequent rerouting grows route_nodes_; this exposes the growth so it can be
  /// watched rather than discovered as memory pressure.
  void route_buffer_stats(size_t& used_slots, size_t& live_slots) const;
  const SegmentGeom& segment(int64_t routing_id) const;

  // --- movement columns ----------------------------------------------------

  size_t size() const { return vehicle_id_.size(); }

  /// Slot holding `vehicle_id`, or -1 if the store does not know it.
  int32_t slot_of(int32_t vehicle_id) const;

  int32_t vehicle_id(int32_t slot) const      { return vehicle_id_[slot]; }
  int32_t start_index(int32_t slot) const     { return start_index_[slot]; }
  double  start_offset_m(int32_t slot) const  { return start_offset_m_[slot]; }
  int64_t time_offset_us(int32_t slot) const  { return time_offset_us_[slot]; }
  int64_t frequency_us(int32_t slot) const    { return frequency_us_[slot]; }
  int64_t fcd_period_us(int32_t slot) const   { return fcd_period_us_[slot]; }
  int32_t active(int32_t slot) const          { return active_[slot]; }
  int32_t route_len(int32_t slot) const       { return route_len_[slot]; }

  void set_position(int32_t slot, int32_t start_index, double start_offset_m);
  void set_time_offset_us(int32_t slot, int64_t time_offset_us);
  void set_active(int32_t slot, int32_t active);

  /// Bulk-refresh the movement columns of `n` vehicles addressed by id.
  ///
  /// This is what keeps a persistent store in step with the Python Vehicle
  /// objects while they are still authoritative: four columns per vehicle, no
  /// route or geometry work. Returns the number updated; an id the store does
  /// not know is a caller error and aborts rather than being skipped, because
  /// silently not updating a vehicle would produce a stale task for it.
  size_t update_movement(const int32_t* vehicle_id,
                         const int32_t* start_index,
                         const double*  start_offset_m,
                         const int64_t* time_offset_us,
                         const int32_t* active,
                         size_t n);

  // --- derived (one array read each, mirroring the Vehicle properties) ------

  /// route_nodes[start_index] -- Vehicle.current_node.
  int64_t current_node(int32_t slot) const;
  /// route_nodes[start_index + 1], or -1 at the end of the route --
  /// Vehicle.next_node.
  int64_t next_node(int32_t slot) const;
  /// Last node of the route -- Vehicle.dest_node.
  int64_t dest_node(int32_t slot) const;

  /// True when the vehicle sits exactly at the end of its current segment, the
  /// condition route.py uses to decide queue membership and a new route.
  bool at_segment_end(int32_t slot) const;

  // --- task construction ---------------------------------------------------

  /// Build the AdvanceTask for one slot. Must match mpi_advance._fill_task
  /// field for field, including the seg1_valid == 0 case at the end of a route.
  AdvanceTask make_task(int32_t slot) const;

  // --- step selection ------------------------------------------------------
  //
  // These mirror Simulation.is_vehicle_within_offset and
  // Simulation.compute_current_offset. Both have a detail that a plain reading
  // of the Python misses; see the notes on the definitions in the .cpp.

  /// Vehicle ids of every active vehicle whose rounded time offset equals
  /// `offset_seconds`, in slot order -- the store's form of
  /// `[v for v in sim.vehicles if sim.is_vehicle_within_offset(v, offset_seconds)]`.
  /// `freq_seconds` is SimSetting._freq_seconds.
  std::vector<int32_t> select_within_offset(int64_t offset_seconds,
                                            int64_t freq_seconds) const;

  /// Smallest NON-ZERO time offset among active vehicles -- the store's form of
  /// compute_current_offset(). Returns false when there is none, matching that
  /// function's `default=None`.
  bool min_active_time_offset_us(int64_t& out_time_offset_us) const;

  /// Number of active vehicles.
  int32_t count_active() const;

  /// Of `vehicle_ids`, those sitting exactly at the end of their current
  /// segment -- the geometric half of singlenode's `need_new_route` filter. The
  /// other half (alternatives != DEFAULT) is a Python-only attribute and stays
  /// in Python.
  std::vector<int32_t> at_segment_end_ids(const std::vector<int32_t>& vehicle_ids) const;

private:
  // One entry per vehicle, indexed by slot.
  std::vector<int32_t> vehicle_id_;
  std::vector<int32_t> start_index_;
  std::vector<double>  start_offset_m_;
  std::vector<int64_t> time_offset_us_;
  std::vector<int64_t> frequency_us_;
  std::vector<int64_t> fcd_period_us_;
  std::vector<int32_t> active_;

  // CSR routes: vehicle `slot` owns route_nodes_[route_offset_[slot] ...
  // + route_len_[slot]) and the matching route_len_[slot] - 1 segment ids.
  std::vector<int32_t> route_offset_;
  std::vector<int32_t> route_len_;
  std::vector<int64_t> route_nodes_;
  std::vector<int64_t> route_segment_ids_;

  std::vector<SegmentGeom> segments_;
  std::unordered_map<int32_t, int32_t> slot_by_vehicle_id_;

  static const SegmentGeom kMissingSegment;

  void fill_segment_desc(SegmentDesc& desc, int64_t routing_id) const;
};

}  // namespace ruthlib
