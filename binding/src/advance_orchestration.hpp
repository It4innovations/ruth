#pragma once

// Master-side driver for the distributed vehicle-advance step. Mirrors
// do_alternatives() in lib.cpp: chunk + scatter phase-1 workloads round-robin,
// wait for completion, then drive the collective FCD/delta exchange.
//
// Per simulation step the master calls, in order:
//   1. advance_drop_old(threshold)           -- prune every GV replica to the
//                                                current offset BEFORE the read,
//                                                matching serial's read-time GV
//   2. do_advance(free_tasks, queues, ...)   -- phase 1: scatter + wait
//   3. sync_advance()                        -- phase 2: broadcast the sync
//                                                workload; every rank enters
//                                                sync_allgatherv and merges FCDs
//   4. get_advance_deltas() / get_advance_fcds()  -- read the results
//
// register_advance_workloads() must be called once from setup_ace().

#include <cstdint>
#include <vector>

#include "advance_types.hpp"

namespace ruthlib {

/// Register the advance workload types with the comm registry. Call once,
/// from setup_ace(), alongside the alternatives workload registrations.
void register_advance_workloads();

/// Phase 1: scatter the free-vehicle chunks and the (whole) queues across ranks
/// and block until every unit has been evaluated. Each rank accumulates its
/// FCDs + deltas locally; nothing is gathered yet.
///
/// @param free_tasks          vehicles not in a queue (order-independent)
/// @param queues              one inner vector per queue, front-to-back; queues
///                            are packed into batches but never split
/// @param departure_epoch_us  simulation departure time, epoch MICROSECONDS
/// @param los_tolerance_s     LoS time tolerance, seconds
///
/// Both families are capped at ranks * ADVANCE_WORK_MULTIPLIER workloads. One
/// workload per queue was the original dispatch and is gone: the queues are
/// 1-3 vehicles, so it made the barrier track message count rather than work.
void do_advance(const std::vector<AdvanceTask>& free_tasks,
                const std::vector<std::vector<AdvanceTask>>& queues,
                int64_t departure_epoch_us,
                double los_tolerance_s);

/// Phase 2: broadcast SyncFCDsWorkload so every rank enters the sync_allgatherv
/// collective, merges the step's FCDs into its GV replica, and (on the master)
/// stashes the full FCD + delta sets. Pruning is NOT done here (see advance_drop_old).
void sync_advance();

/// Prune every rank's GV replica to datetime >= threshold_epoch_s. Call at the
/// START of each step (before do_advance), at departure + current_offset, to
/// reproduce serial's read-time GV contents. Passing -inf prunes nothing.
void advance_drop_old(double threshold_epoch_s);

/// Wall time, in milliseconds, of the master-side phases of one step.
///
/// Instrumentation only. `advance_vehicle` is one timer wrapping nine things and
/// the interesting question -- is a slow step spent MARSHALLING work into ACE
/// messages, WAITING at the barrier for a straggler rank, or in the two
/// all-gathers -- cannot be answered without separating them. Those three have
/// nothing in common as fixes.
struct AdvancePhaseTimes {
  double drop_old_ms = 0.0;      ///< broadcast + wait for the GV prune
  double dispatch_ms = 0.0;      ///< building and async_send-ing the workloads
  double barrier_ms = 0.0;       ///< drain_until_quiescent: the actual makespan
  double sync_ms = 0.0;          ///< broadcast + wait for the FCD/delta gather
  int32_t free_chunks = 0;       ///< workloads sent for free vehicles
  int32_t queue_workloads = 0;   ///< MESSAGES sent for queues (batched or not)
  int32_t queue_count = 0;       ///< queues in the step, however they were packed
  int32_t longest_queue = 0;     ///< vehicles in the largest single queue
};

/// Phase times recorded by the last do_advance / sync_advance / drop_old.
AdvancePhaseTimes get_advance_phase_times();

/// Append FCDs to THIS rank's local accumulator before sync_advance, so they
/// are included in the gather. Used by the master to inject the stationary FCDs
/// of "after-gap" waiting vehicles (queued+selected behind a non-selected
/// vehicle) that are not evaluated by any workload.
void advance_add_local_fcds(const double* datetime_s, const int32_t* vehicle_id,
                            const int64_t* segment_id, const double* offset_m,
                            const double* speed_mps, size_t n);

// --- Results, as struct-of-arrays for zero-copy numpy in the bindings -------

struct AdvanceDeltasFlat {
  std::vector<int32_t> vehicle_id;
  std::vector<int32_t> new_start_index;
  std::vector<double>  new_offset_m;
  std::vector<int64_t> new_time_offset_us;
  std::vector<int32_t> moved;
  std::vector<int32_t> changed_segment;
  std::vector<int32_t> reached_end;
};

struct AdvanceFCDsFlat {
  std::vector<double>  datetime_seconds;
  std::vector<int32_t> vehicle_id;
  std::vector<int64_t> segment_id;
  std::vector<double>  offset_from_start;
  std::vector<double>  vehicle_speed_mps;
};

/// Master reads the deltas gathered by the last sync_advance (empties the
/// buffer). Used to update Vehicle objects, replay queue bookkeeping, and
/// (with the FCDs) build the history.
AdvanceDeltasFlat get_advance_deltas();

/// Master reads the FCDs gathered by the last sync_advance (empties the
/// buffer). Used to build the FCD history.
AdvanceFCDsFlat get_advance_fcds();

}  // namespace ruthlib
