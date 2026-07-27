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

// Determinism aid. The gathered FCDs are a set-identical but arbitrarily-ordered
// concatenation of every rank's contribution, so the FCD history file's row
// order varies with rank count (results are unaffected — same records). In
// Debug / RelWithDebInfo builds CMake defines RUTH_ADVANCE_SORT_FCDS; the Python
// master then reorders each step's FCDs into serial's write order
// (free vehicles, then queues FIFO, then waiting; datetime within each vehicle),
// making the FCD history byte-identical to a serial run. Reordering happens
// entirely master-side — the workloads and the serial path are untouched.
// Release leaves the gather order (avoids the per-step reorder).
#ifdef RUTH_ADVANCE_SORT_FCDS
static constexpr bool kSortFcdsForDeterminism = true;
#else
static constexpr bool kSortFcdsForDeterminism = false;
#endif

bool advance_debug_sort_enabled() { return kSortFcdsForDeterminism; }

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

  std::vector<std::shared_ptr<ace::workload>> handles;
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
    handles.push_back(wl);
  }

  // Queues: one whole queue per workload (never split across ranks).
  for (const auto& queue : queues) {
    if (queue.empty()) continue;
    auto wl = std::make_shared<QueueWorkload>(queue, departure_epoch_us, los_tolerance_s);
    ++dispatch_index;
    comm->async_send(wl, dispatch_index % num_ranks);
    handles.push_back(wl);
  }

  // Barrier via completion tracking: block until every phase-1 unit is done.
  for (auto& h : handles) h->wait();
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
