#pragma once

// ACE workloads for the distributed vehicle-advance model.
//
// Two phase-1 families, mirroring ruth/simulator/route.py:
//   * FreeVehicleWorkload : vehicles NOT in a queue. Order-independent,
//                           evaluated with ace::parallelfor over the chunk.
//   * QueueWorkload       : one WHOLE queue per unit. Evaluated SEQUENTIALLY
//                           (FIFO, blocked-front-blocks-followers); parallelism
//                           comes only from having many queues.
//
// Result flow: both the FCD samples and the per-vehicle deltas are accumulated
// rank-locally during phase 1, then exchanged in a single per-step
// sync_allgatherv driven by SyncFCDsWorkload. Every rank slices the gathered
// FCDs into its GV replica; the master additionally slices the deltas to update
// its Vehicle objects, replay queue bookkeeping, and build the FCD history.

#include <memory>
#include <vector>

#include "workload/workload.hpp"

#include "advance_types.hpp"

namespace ruthlib {

/// Evaluate a batch of AdvanceTasks, accumulating both the FCD samples and the
/// per-vehicle deltas into AdvanceState (they are exchanged later by the
/// sync_allgatherv). `sequential` selects queue semantics (FIFO, stop at the
/// first vehicle that does not move); otherwise the batch runs in parallel.
void evaluate_batch(const std::vector<AdvanceTask>& tasks,
                    int64_t departure_epoch_us,
                    double los_tolerance_s,
                    bool sequential);

// --- Free vehicles: order-independent, parallelfor over the chunk ----------
class FreeVehicleWorkload : public ace::workload {
public:
  std::vector<AdvanceTask> tasks;
  int64_t departure_epoch_us = 0;
  double los_tolerance_s = 0.0;

  FreeVehicleWorkload();
  FreeVehicleWorkload(std::vector<AdvanceTask> tasks, int64_t departure_epoch_us, double los_tolerance_s);
  FreeVehicleWorkload(const std::shared_ptr<char> data, size_t size);

  void execute() override;
  size_t size() const override;
  char* marshal() const override;
};

// --- Queue vehicles: one whole queue, FIFO, sequential ---------------------
class QueueWorkload : public ace::workload {
public:
  std::vector<AdvanceTask> tasks;   // one queue, front-to-back order
  int64_t departure_epoch_us = 0;
  double los_tolerance_s = 0.0;

  QueueWorkload();
  QueueWorkload(std::vector<AdvanceTask> tasks, int64_t departure_epoch_us, double los_tolerance_s);
  QueueWorkload(const std::shared_ptr<char> data, size_t size);

  void execute() override;
  size_t size() const override;
  char* marshal() const override;
};

// --- Broadcast trigger: every rank enters the sync_allgatherv collective ----
// Broadcast by the master AFTER all phase-1 workloads have completed. Gathers
// each rank's FCD samples + deltas so the step's FCDs converge into every GV
// replica (and the master receives every delta) before the next step.
// NOTE: this no longer prunes the GV. Pruning moved to DropOldWorkload, which
// runs BEFORE do_advance reads the replica, so the read-time GV contents match
// serial (which prunes at end of the previous step with the current offset).
class SyncFCDsWorkload : public ace::workload {
public:
  SyncFCDsWorkload();
  SyncFCDsWorkload(const std::shared_ptr<char> data, size_t size);

  void execute() override;
  size_t size() const override;
  char* marshal() const override;
  ace::workload_types get_type() const override;
};

// --- Broadcast: prune every rank's GlobalView replica to datetime >= threshold.
// Issued at the START of each step (before do_advance reads), at the same
// threshold serial uses (departure + current offset). This is what makes the
// distributed LoS match serial: the prune threshold sits inside the LoS
// tolerance window, so pruning one round too late (as the old in-sync prune did)
// leaves stale vehicles counted as "ahead" and diverges the trajectories.
class DropOldWorkload : public ace::workload {
public:
  double threshold_epoch_s;   // drop GV records with datetime < this

  DropOldWorkload();
  explicit DropOldWorkload(double threshold_epoch_s);
  DropOldWorkload(const std::shared_ptr<char> data, size_t size);

  void execute() override;
  size_t size() const override;
  char* marshal() const override;
  ace::workload_types get_type() const override;
};

}  // namespace ruthlib
