#pragma once

// C++ port of ruth/simulator/queues.py QueuesManager.
//
// A queue holds the ids of the vehicles waiting at the end of one segment,
// keyed by that segment's (node_from, node_to). Vehicles are staged in and out
// during delta replay and the staged operations are applied together by
// batch_update(), exactly as the Python does.
//
// SHADOW ONLY FOR NOW. The Python QueuesManager remains authoritative; this is
// fed the same operations and required to reach the same state, so that the
// switchover has evidence behind it rather than hope.
//
// TWO THINGS THE PYTHON DOES THAT ARE EASY TO MISS, both reproduced here:
//
//   * `queues` is a defaultdict(deque), so READING a key CREATES it. The lookup
//     in _split_vehicles is a read, and it is what brings most keys into
//     existence -- see touch().
//   * the ORDER those keys were first created in is observable: _split_vehicles
//     iterates queues.items() to build the queue groups, and that order decides
//     the order work is dispatched, deltas are replayed, and FCDs are written.
//     A std::unordered_map alone would lose it, so insertion order is kept
//     explicitly.

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace ruthlib {

class QueueRegistry {
public:
  static QueueRegistry& instance();

  QueueRegistry() = default;
  QueueRegistry(const QueueRegistry&) = delete;
  QueueRegistry& operator=(const QueueRegistry&) = delete;

  void clear();

  /// Get (creating if absent) the queue for one segment -- defaultdict
  /// semantics. Creation order is recorded; this is how key order is
  /// established, so it must be called in the same order Python reads them.
  std::deque<int32_t>& touch(int64_t node_from, int64_t node_to);

  /// Is `vehicle_id` in that segment's queue? Does NOT create the key, so a
  /// membership test cannot perturb the ordering.
  bool contains(int64_t node_from, int64_t node_to, int32_t vehicle_id) const;

  void stage_add(int32_t vehicle_id, int64_t node_from, int64_t node_to);
  void stage_remove(int32_t vehicle_id, int64_t node_from, int64_t node_to);

  /// Apply the staged operations: removals first (each pops the FRONT of its
  /// queue and must find the vehicle it named there), then additions at the
  /// back. Mirrors QueuesManager.batch_update, asserts included.
  void batch_update();

  size_t staged_removals() const { return to_be_removed_.size(); }
  size_t staged_additions() const { return to_be_added_.size(); }

  /// Number of keys, including empty queues -- an empty deque left behind by a
  /// lookup is still a key, and Python counts it too.
  size_t key_count() const { return order_.size(); }

  /// Flatten for comparison against Python: keys in creation order, plus a CSR
  /// of the ids queued under each.
  void snapshot(std::vector<int64_t>& node_from,
                std::vector<int64_t>& node_to,
                std::vector<int32_t>& offsets,
                std::vector<int32_t>& vehicle_ids) const;

private:
  struct Key {
    int64_t node_from;
    int64_t node_to;
    bool operator==(const Key& other) const {
      return node_from == other.node_from && node_to == other.node_to;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const {
      // The node ids are OSM ids, sparse and large; mixing keeps them off one
      // bucket without pretending to be a strong hash.
      const uint64_t a = static_cast<uint64_t>(k.node_from);
      const uint64_t b = static_cast<uint64_t>(k.node_to);
      return static_cast<size_t>(a * 0x9E3779B97F4A7C15ull ^ (b + 0x165667B1ull + (a << 6)));
    }
  };
  struct StagedOp {
    int32_t vehicle_id;
    int64_t node_from;
    int64_t node_to;
  };

  std::unordered_map<Key, size_t, KeyHash> index_;  ///< key -> position in order_
  std::vector<Key> order_;                          ///< creation order
  std::vector<std::deque<int32_t>> queues_;         ///< parallel to order_
  std::vector<StagedOp> to_be_added_;
  std::vector<StagedOp> to_be_removed_;

  size_t slot_for(int64_t node_from, int64_t node_to);
};

}  // namespace ruthlib
