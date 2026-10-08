#include "advance_orchestration.hpp"

#include <algorithm>
#include <memory>

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

void register_advance_workloads() {
  auto comm = ace::CommInterface::instance();
  comm->register_workload<FreeVehicleWorkload>();
  comm->register_workload<QueueWorkload>();
  comm->register_workload<SyncFCDsWorkload>();
  comm->register_workload<DropOldWorkload>();
}

void advance_drop_old(double threshold_epoch_s) {
  auto comm = ace::CommInterface::instance();
  auto wl = std::make_shared<DropOldWorkload>(threshold_epoch_s);
  // Prune every rank's GV replica before do_advance reads it, so the read-time
  // contents match serial's (which prunes at the end of the previous step).
  comm->async_broadcast(wl);
  wl->wait();
}

void do_advance(const std::vector<AdvanceTask>& free_tasks,
                const std::vector<std::vector<AdvanceTask>>& queues,
                int64_t departure_epoch_us,
                double los_tolerance_s) {
  auto comm = ace::CommInterface::instance();
  const int num_ranks = comm->get_size();

  size_t dispatch_index = 0;

  // Free vehicles: split into chunks and scatter round-robin. dispatch_index is
  // incremented before the modulo (as in do_alternatives) so the first chunk
  // goes to rank 1 and the busy master picks up work last.
  const size_t work_size =
      std::max<size_t>(1, free_tasks.size() / (num_ranks * ADVANCE_WORK_MULTIPLIER));
  for (size_t start = 0; start < free_tasks.size(); start += work_size) {
    const size_t end = std::min(start + work_size, free_tasks.size());
    std::vector<AdvanceTask> chunk(free_tasks.begin() + start, free_tasks.begin() + end);
    auto wl = std::make_shared<FreeVehicleWorkload>(std::move(chunk),
                                                    departure_epoch_us, los_tolerance_s);
    ++dispatch_index;
    comm->async_send(wl, dispatch_index % num_ranks);
  }

  // Queues: one whole queue per workload (never split across ranks).
  for (const auto& queue : queues) {
    if (queue.empty()) continue;
    auto wl = std::make_shared<QueueWorkload>(queue, departure_epoch_us, los_tolerance_s);
    ++dispatch_index;
    comm->async_send(wl, dispatch_index % num_ranks);
  }

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
}

void sync_advance() {
  auto comm = ace::CommInterface::instance();
  auto wl = std::make_shared<SyncFCDsWorkload>();
  // Broadcast reaches every rank (including the master), so all ranks enter the
  // sync_allgatherv collective inside SyncFCDsWorkload::execute().
  comm->async_broadcast(wl);
  wl->wait();
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
