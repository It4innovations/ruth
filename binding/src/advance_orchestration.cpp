#include "advance_orchestration.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <utility>

#include "comm/comm_interface.hpp"
#include "workload/workload.hpp"

#include "advance_state.hpp"
#include "advance_workloads.hpp"

namespace ruthlib {

// Oversubscription factor: how many chunks per rank to create, so the scheduler
// has slack to balance load. Matches MULTIPLIER in lib.cpp.
static constexpr size_t ADVANCE_WORK_MULTIPLIER = 16;

// NOTE: the FCD reordering that used to be selected here by the compile-time
// RUTH_ADVANCE_SORT_FCDS flag is now a runtime setting, config.json
// ace.deterministic-order, read by ruth/simulator/mpi_advance.py. The flag, the
// kSortFcdsForDeterminism constant and advance_debug_sort_enabled() are gone;
// reordering is still entirely master-side and the workloads are untouched.

namespace {

// Phase times for the step in progress. Master-only, written from the master
// thread between collectives, so no synchronisation is needed.
AdvancePhaseTimes g_phase_times;

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& since) {
  return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

}  // namespace

AdvancePhaseTimes get_advance_phase_times() { return g_phase_times; }

void register_advance_workloads() {
  auto comm = ace::CommInterface::instance();
  comm->register_workload<FreeVehicleWorkload>();
  comm->register_workload<QueueWorkload>();
  comm->register_workload<QueueBatchWorkload>();
  comm->register_workload<SyncFCDsWorkload>();
  comm->register_workload<DropOldWorkload>();
}

void advance_drop_old(double threshold_epoch_s) {
  const auto started = Clock::now();

  auto comm = ace::CommInterface::instance();
  auto wl = std::make_shared<DropOldWorkload>(threshold_epoch_s);
  // Prune every rank's GV replica before do_advance reads it, so the read-time
  // contents match serial's (which prunes at the end of the previous step).
  comm->async_broadcast(wl);
  wl->wait();

  // First phase of a step, so this is where the counters reset.
  g_phase_times = AdvancePhaseTimes{};
  g_phase_times.drop_old_ms = elapsed_ms(started);
}

void do_advance(const std::vector<AdvanceTask>& free_tasks,
                const std::vector<std::vector<AdvanceTask>>& queues,
                int64_t departure_epoch_us,
                double los_tolerance_s,
                bool batch_workloads) {
  const auto dispatch_started = Clock::now();

  auto comm = ace::CommInterface::instance();
  const int num_ranks = comm->get_size();

  size_t dispatch_index = 0;

  // How many workloads we want in flight, in total, per family. Oversubscribed
  // so the scheduler still has slack to steal with.
  const size_t target_workloads =
      std::max<size_t>(1, static_cast<size_t>(num_ranks) * ADVANCE_WORK_MULTIPLIER);

  // Free vehicles: split into chunks and scatter round-robin. dispatch_index is
  // incremented before the modulo (as in do_alternatives) so the first chunk
  // goes to rank 1 and the busy master picks up work last.
  //
  // The chunk size CEILS n/target so the COUNT is capped. The old form floored
  // it and clamped to 1, which inverts the intent: below ranks*16 tasks the
  // size collapsed to 1 and every vehicle got its own workload. At 32 ranks
  // that is any step with fewer than 512 free vehicles -- i.e. most of them.
  const size_t work_size =
      batch_workloads
          ? std::max<size_t>(1, (free_tasks.size() + target_workloads - 1) / target_workloads)
          : std::max<size_t>(1, free_tasks.size() / (num_ranks * ADVANCE_WORK_MULTIPLIER));
  for (size_t start = 0; start < free_tasks.size(); start += work_size) {
    const size_t end = std::min(start + work_size, free_tasks.size());
    std::vector<AdvanceTask> chunk(free_tasks.begin() + start, free_tasks.begin() + end);
    auto wl = std::make_shared<FreeVehicleWorkload>(std::move(chunk),
                                                    departure_epoch_us, los_tolerance_s);
    ++dispatch_index;
    comm->async_send(wl, dispatch_index % num_ranks);
  }

  g_phase_times.free_chunks = static_cast<int32_t>(dispatch_index);

  size_t total_queued = 0;
  for (const auto& queue : queues) {
    if (queue.empty()) continue;
    total_queued += queue.size();
    ++g_phase_times.queue_count;
    g_phase_times.longest_queue =
        std::max(g_phase_times.longest_queue, static_cast<int32_t>(queue.size()));
  }

  if (batch_workloads) {
    // Pack whole queues into at most target_workloads messages. Queues are
    // never split -- the FIFO rule lives inside one -- they are only shipped
    // together. Filling by vehicle count rather than by queue count keeps the
    // batches even when queue lengths vary.
    const size_t per_batch =
        std::max<size_t>(1, (total_queued + target_workloads - 1) / target_workloads);

    std::vector<AdvanceTask> batch_tasks;
    std::vector<int32_t> batch_offsets{0};

    // std::exchange, not std::move: handing the buffers over and starting fresh
    // ones is ONE expression, so there is no window in which batch_tasks is a
    // moved-from vector being relied on to behave. offsets restart at {0}
    // because the CSR of every batch is zero-based.
    auto flush_batch = [&]() {
      if (batch_offsets.size() < 2) {
        return;  // nothing accumulated since the last flush
      }
      auto wl = std::make_shared<QueueBatchWorkload>(
          std::exchange(batch_tasks, std::vector<AdvanceTask>{}),
          std::exchange(batch_offsets, std::vector<int32_t>{0}),
          departure_epoch_us, los_tolerance_s);
      ++dispatch_index;
      comm->async_send(wl, dispatch_index % num_ranks);
      ++g_phase_times.queue_workloads;
    };

    for (const auto& queue : queues) {
      if (queue.empty()) continue;
      batch_tasks.insert(batch_tasks.end(), queue.begin(), queue.end());
      batch_offsets.push_back(static_cast<int32_t>(batch_tasks.size()));
      if (batch_tasks.size() >= per_batch) {
        flush_batch();
      }
    }
    flush_batch();
  } else {
    // Legacy: one whole queue per workload (never split across ranks).
    for (const auto& queue : queues) {
      if (queue.empty()) continue;
      auto wl = std::make_shared<QueueWorkload>(queue, departure_epoch_us, los_tolerance_s);
      ++dispatch_index;
      comm->async_send(wl, dispatch_index % num_ranks);
      ++g_phase_times.queue_workloads;
    }
  }

  g_phase_times.dispatch_ms = elapsed_ms(dispatch_started);
  const auto barrier_started = Clock::now();

  // No local handles kept: async_send() marshals each workload into its own send
  // buffer and registers the shared_ptr in the scheduler's waiting map, which owns
  // it until the completion ack — so `wl` may die at the end of each iteration.
  // Fire-and-forget instead of waiting per handle. This collective barrier drains
  // every dispatched movement workload's ack across all ranks AND runs ACE's
  // work-stealer so per-vehicle chunks rebalance off busy ranks (load balancing).
  // Requires every rank at the barrier (workers are in run_worker_loop()). Issued
  // unconditionally (even with zero tasks) to keep the per-step barrier count
  // deterministic. The barrier is here in the dispatcher, NOT inside FreeVehicle/
  // QueueWorkload::execute() (those run on scheduler threads and must not call a
  // collective).
  comm->barrier();
  g_phase_times.barrier_ms = elapsed_ms(barrier_started);
}

void sync_advance() {
  const auto started = Clock::now();

  auto comm = ace::CommInterface::instance();
  auto wl = std::make_shared<SyncFCDsWorkload>();
  // Broadcast reaches every rank (including the master), so all ranks enter the
  // sync_allgatherv collective inside SyncFCDsWorkload::execute().
  comm->async_broadcast(wl);
  wl->wait();

  g_phase_times.sync_ms = elapsed_ms(started);
}

void advance_add_local_fcds(const double* datetime_s, const int32_t* vehicle_id,
                            const int64_t* segment_id, const double* offset_m,
                            const double* speed_mps, size_t n) {
  std::vector<FCDRecord> fcds;
  fcds.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    fcds.emplace_back(datetime_s[i], vehicle_id[i], segment_id[i], offset_m[i], speed_mps[i]);
  }
  AdvanceState::instance().append_fcds(fcds);
}

AdvanceDeltasFlat get_advance_deltas() {
  std::vector<AdvanceDelta> deltas = AdvanceState::instance().take_last_synced_deltas();

  AdvanceDeltasFlat out;
  const size_t n = deltas.size();
  out.vehicle_id.reserve(n);
  out.new_start_index.reserve(n);
  out.new_offset_m.reserve(n);
  out.new_time_offset_us.reserve(n);
  out.moved.reserve(n);
  out.changed_segment.reserve(n);
  out.reached_end.reserve(n);

  for (const auto& d : deltas) {
    out.vehicle_id.push_back(d.vehicle_id);
    out.new_start_index.push_back(d.new_start_index);
    out.new_offset_m.push_back(d.new_offset_m);
    out.new_time_offset_us.push_back(d.new_time_offset_us);
    out.moved.push_back(d.moved);
    out.changed_segment.push_back(d.changed_segment);
    out.reached_end.push_back(d.reached_end);
  }
  return out;
}

AdvanceFCDsFlat get_advance_fcds() {
  std::vector<FCDRecord> fcds = AdvanceState::instance().take_last_synced_fcds();

  AdvanceFCDsFlat out;
  const size_t n = fcds.size();
  out.datetime_seconds.reserve(n);
  out.vehicle_id.reserve(n);
  out.segment_id.reserve(n);
  out.offset_from_start.reserve(n);
  out.vehicle_speed_mps.reserve(n);

  for (const auto& f : fcds) {
    out.datetime_seconds.push_back(f.datetime_seconds);
    out.vehicle_id.push_back(f.vehicle_id);
    out.segment_id.push_back(f.segment_id);
    out.offset_from_start.push_back(f.offset_from_start);
    out.vehicle_speed_mps.push_back(f.vehicle_speed_mps);
  }
  return out;
}

}  // namespace ruthlib
