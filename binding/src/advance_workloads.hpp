#pragma once

// ACE workloads for the distributed vehicle-advance model.
//
// Two phase-1 families, mirroring ruth/simulator/route.py:
//   * FreeVehicleWorkload : vehicles NOT in a queue. Order-independent,
//                           evaluated with ace::parallelfor over the chunk.
//   * QueueBatchWorkload  : MANY whole queues per unit. Parallel across the
//                           queues it carries, sequential within each one
//                           (FIFO, blocked-front-blocks-followers). A queue is
//                           packed with others but never split.
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

/// Evaluate a batch of INDEPENDENT AdvanceTasks in parallel, accumulating both
/// the FCD samples and the per-vehicle deltas into AdvanceState (they are
/// exchanged later by the sync_allgatherv). Queue semantics -- FIFO, stop at the
/// first vehicle that does not move -- are not here: they belong to one queue,
/// and evaluate_one_queue implements them.
void evaluate_batch(const std::vector<AdvanceTask>& tasks,
                    int64_t departure_epoch_us,
                    double los_tolerance_s);

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

/// Walk ONE queue: tasks[begin .. end), front to back, stopping at the first
/// vehicle that does not move; everything behind it waits. Writes each delta at
/// its flat index and appends that queue's FCDs to `out_fcds`.
///
/// Split out so it can be exercised directly, and so the per-queue rule lives
/// in exactly one place.
void evaluate_one_queue(const std::vector<AdvanceTask>& tasks,
                        int32_t begin, int32_t end,
                        const GlobalView& gv,
                        int64_t departure_epoch_us,
                        double los_tolerance_s,
                        std::vector<AdvanceDelta>& deltas,
                        std::vector<FCDRecord>& out_fcds);

/// Evaluate MANY queues in one go. `offsets` is a CSR over `tasks`: queue q is
/// tasks[offsets[q] .. offsets[q+1]). Queues run in PARALLEL across the rank's
/// worker threads; each queue is walked sequentially, because the FIFO rule is
/// a property of one queue and queues are independent of each other.
void evaluate_queue_batch(const std::vector<AdvanceTask>& tasks,
                          const std::vector<int32_t>& offsets,
                          int64_t departure_epoch_us,
                          double los_tolerance_s);

// --- Many queues in one workload -------------------------------------------
//
// WHY THIS EXISTS. do_advance used to send one workload per queue, and the
// queues are tiny -- measured at 1.2 to 2.7 vehicles each, with up to 14,010 of
// them per step. Every one cost its own make_shared, marshalled send buffer,
// MPI message, scheduler waiting-map entry and notify_workload ack: about
// 0.4 ms of overhead around roughly 24-60 us of kinematics. The barrier was
// 93-97% of the advance phase, and it tracked the WORKLOAD COUNT, not the
// vehicle count.
//
// Batching changes nothing about semantics. Queues are independent of one
// another -- the FIFO rule and its blocked prefix apply within a queue -- so
// packing several into one message only changes how they are shipped. Each
// vehicle still appears in exactly one queue, so its FCD subsequence is
// unaffected too.
class QueueBatchWorkload : public ace::workload {
public:
  std::vector<AdvanceTask> tasks;    // every queue's vehicles, concatenated
  std::vector<int32_t> offsets;      // CSR delimiters, n_queues + 1
  int64_t departure_epoch_us = 0;
  double los_tolerance_s = 0.0;

  QueueBatchWorkload();
  QueueBatchWorkload(std::vector<AdvanceTask> tasks, std::vector<int32_t> offsets,
                     int64_t departure_epoch_us, double los_tolerance_s);
  QueueBatchWorkload(const std::shared_ptr<char> data, size_t size);

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
