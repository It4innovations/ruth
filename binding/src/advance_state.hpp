#pragma once

// Per-rank state for the distributed vehicle-advance model: the GlobalView
// replica used for LoS queries plus the per-step accumulators. Kept separate
// from lib.cpp's `State` (which owns the routing graph for the alternatives
// algorithm); this singleton owns the FCD / level-of-service side of the world.

#include <mutex>
#include <vector>

#include "advance_types.hpp"
#include "globalview.hpp"   // FCDRecord, GlobalView

namespace ruthlib {

class AdvanceState {
public:
  static AdvanceState& instance();

  AdvanceState(const AdvanceState&) = delete;
  AdvanceState& operator=(const AdvanceState&) = delete;

  GlobalView& gv();

  double los_tolerance_s() const;
  void set_los_tolerance_s(double t);

  // --- per-step accumulators (thread-safe) ---
  void reset_step();
  void append_fcds(const std::vector<FCDRecord>& fcds);
  void append_delta(const AdvanceDelta& d);

  /// This rank's locally produced FCDs / deltas, moved out for sync_allgatherv.
  std::vector<FCDRecord> take_local_fcds();
  std::vector<AdvanceDelta> take_local_deltas();

  // Full FCD + delta sets from the last sync (held by every rank after
  // sync_allgatherv); the master reads them to build the FCD history and to
  // update its Vehicle objects and queues.
  void set_last_synced_fcds(std::vector<FCDRecord>&& f);
  void set_last_synced_deltas(std::vector<AdvanceDelta>&& d);
  std::vector<FCDRecord> take_last_synced_fcds();
  std::vector<AdvanceDelta> take_last_synced_deltas();

private:
  AdvanceState() = default;

  GlobalView gv_;
  double los_tolerance_s_ = 0.0;
  std::mutex mtx_;
  std::vector<FCDRecord> step_fcds_;
  std::vector<AdvanceDelta> step_deltas_;
  std::vector<FCDRecord> last_synced_fcds_;
  std::vector<AdvanceDelta> last_synced_deltas_;
};

}  // namespace ruthlib
