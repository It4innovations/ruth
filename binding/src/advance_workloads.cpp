#include "advance_workloads.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <thread>

#include "comm/comm_interface.hpp"
#include "workload/parallelfor.hpp"

#include "advance_kernel.hpp"
#include "advance_marshal.hpp"
#include "advance_state.hpp"

namespace ruthlib {

// ---------------------------------------------------------------------------
// evaluate_batch
// ---------------------------------------------------------------------------
void evaluate_batch(const std::vector<AdvanceTask>& tasks,
                    int64_t departure_epoch_us,
                    double los_tolerance_s,
                    bool sequential) {
  auto& st = AdvanceState::instance();
  const GlobalView& gv = st.gv();

  std::vector<AdvanceDelta> deltas;
  deltas.reserve(tasks.size());
  std::vector<FCDRecord> fcds;

  if (sequential) {
    // Queue semantics: FIFO, stop at the first vehicle that does not move.
    // Vehicles behind a non-mover simply wait (advance time, stationary FCD),
    // matching route.py:advance_waiting_vehicle.
    bool blocked = false;
    for (const auto& t : tasks) {
      if (!blocked) {
        AdvanceDelta d = advance_one_vehicle(t, gv, departure_epoch_us, los_tolerance_s, fcds);
        deltas.push_back(d);
        if (!d.moved) {
          blocked = true;  // this vehicle blocks the rest of the queue
        }
      } else {
        const int64_t current_epoch_us = departure_epoch_us + t.time_offset_us;
        const int64_t end_epoch_us = current_epoch_us + t.frequency_us;
        fcds.emplace_back(static_cast<double>(end_epoch_us) / 1e6,
                          t.vehicle_id, t.seg0.segment_id, t.start_offset_m, 0.0);
        AdvanceDelta d{};
        d.vehicle_id = t.vehicle_id;
        d.new_start_index = t.start_index;
        d.new_offset_m = t.start_offset_m;
        d.new_time_offset_us = t.time_offset_us + t.frequency_us;
        d.moved = 0;
        d.changed_segment = 0;
        d.reached_end = 0;
        deltas.push_back(d);
      }
    }
  } else {
    // Free vehicles: fully parallel. Per-index FCD buffers merged at the end to
    // avoid contention; deltas indexed by position to keep order stable.
    const size_t n = tasks.size();
    deltas.resize(n);
    std::vector<std::vector<FCDRecord>> per_index(n);

    const size_t workers = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t chunk = std::max<size_t>(1, n / (workers * 16));
    ace::parallelfor(
        ace::Range<size_t>(0, n, chunk),
        [&](const size_t& i) {
          std::vector<FCDRecord> local;
          deltas[i] = advance_one_vehicle(tasks[i], gv, departure_epoch_us, los_tolerance_s, local);
          per_index[i] = std::move(local);
        },
        true);

    size_t total = 0;
    for (const auto& v : per_index) total += v.size();
    fcds.reserve(total);
    for (auto& v : per_index) fcds.insert(fcds.end(), v.begin(), v.end());
  }

  // Accumulate rank-locally; both are exchanged later by the sync_allgatherv.
  st.append_fcds(fcds);
  for (const auto& d : deltas) st.append_delta(d);
}

// ---------------------------------------------------------------------------
// FreeVehicleWorkload
// ---------------------------------------------------------------------------
FreeVehicleWorkload::FreeVehicleWorkload() : workload() {}

FreeVehicleWorkload::FreeVehicleWorkload(std::vector<AdvanceTask> tasks,
                                         int64_t departure_epoch_us, double los_tolerance_s)
    : workload(), tasks(std::move(tasks)),
      departure_epoch_us(departure_epoch_us), los_tolerance_s(los_tolerance_s) {}

FreeVehicleWorkload::FreeVehicleWorkload(const std::shared_ptr<char> data, size_t size)
    : workload(data, size) {
  const char* p = data.get();
  std::memcpy(&departure_epoch_us, p, sizeof(int64_t)); p += sizeof(int64_t);
  std::memcpy(&los_tolerance_s, p, sizeof(double)); p += sizeof(double);
  tasks = get_pod_vec<AdvanceTask>(p);
}

void FreeVehicleWorkload::execute() {
  evaluate_batch(tasks, departure_epoch_us, los_tolerance_s, /*sequential=*/false);
}

size_t FreeVehicleWorkload::size() const {
  return sizeof(int64_t) + sizeof(double) + pod_vec_size(tasks);
}

char* FreeVehicleWorkload::marshal() const {
  char* data = workload::marshal();
  char* p = data;
  std::memcpy(p, &departure_epoch_us, sizeof(int64_t)); p += sizeof(int64_t);
  std::memcpy(p, &los_tolerance_s, sizeof(double)); p += sizeof(double);
  put_pod_vec(p, tasks);
  return data;
}

// ---------------------------------------------------------------------------
// QueueWorkload
// ---------------------------------------------------------------------------
QueueWorkload::QueueWorkload() : workload() {}

QueueWorkload::QueueWorkload(std::vector<AdvanceTask> tasks,
                             int64_t departure_epoch_us, double los_tolerance_s)
    : workload(), tasks(std::move(tasks)),
      departure_epoch_us(departure_epoch_us), los_tolerance_s(los_tolerance_s) {}

QueueWorkload::QueueWorkload(const std::shared_ptr<char> data, size_t size)
    : workload(data, size) {
  const char* p = data.get();
  std::memcpy(&departure_epoch_us, p, sizeof(int64_t)); p += sizeof(int64_t);
  std::memcpy(&los_tolerance_s, p, sizeof(double)); p += sizeof(double);
  tasks = get_pod_vec<AdvanceTask>(p);
}

void QueueWorkload::execute() {
  evaluate_batch(tasks, departure_epoch_us, los_tolerance_s, /*sequential=*/true);
}

size_t QueueWorkload::size() const {
  return sizeof(int64_t) + sizeof(double) + pod_vec_size(tasks);
}

char* QueueWorkload::marshal() const {
  char* data = workload::marshal();
  char* p = data;
  std::memcpy(p, &departure_epoch_us, sizeof(int64_t)); p += sizeof(int64_t);
  std::memcpy(p, &los_tolerance_s, sizeof(double)); p += sizeof(double);
  put_pod_vec(p, tasks);
  return data;
}

// ---------------------------------------------------------------------------
// SyncFCDsWorkload
// ---------------------------------------------------------------------------
namespace {

// All-gather a POD vector across all ranks. The element count is derived from
// the gathered byte count, so no in-band framing is needed. COLLECTIVE: every
// rank must call this the same number of times, in the same order.
template <typename T>
std::vector<T> gather_pod_vector(const std::vector<T>& local) {
  std::vector<char> gathered = ace::CommInterface::instance()->sync_allgatherv(
      reinterpret_cast<const char*>(local.data()), local.size() * sizeof(T));
  const size_t n = gathered.size() / sizeof(T);
  std::vector<T> out(n);
  if (n) std::memcpy(out.data(), gathered.data(), n * sizeof(T));
  return out;
}

}  // namespace

SyncFCDsWorkload::SyncFCDsWorkload() : workload() {}

SyncFCDsWorkload::SyncFCDsWorkload(const std::shared_ptr<char> data, size_t size)
    : workload(data, size) {}

void SyncFCDsWorkload::execute() {
  auto& st = AdvanceState::instance();

  // Two independent collectives, one per record type. Every rank issues them in
  // the same order, so the two MPI_Allgatherv calls pair up correctly. Each
  // returns that type's records from every rank, concatenated in rank order
  // (order-independent for GV aggregation, so rank order does not matter).
  std::vector<FCDRecord> all_fcds = gather_pod_vector(st.take_local_fcds());
  std::vector<AdvanceDelta> all_deltas = gather_pod_vector(st.take_local_deltas());

  // Every rank integrates the FCDs into its GV replica. No pruning here — the
  // GV is pruned by DropOldWorkload at the START of the step (before the read).
  st.gv().add_batch(all_fcds);

  // Stash the full sets for the master to read (FCD history + vehicle/queue
  // updates). Non-master ranks simply overwrite these next step.
  st.set_last_synced_fcds(std::move(all_fcds));
  st.set_last_synced_deltas(std::move(all_deltas));
}

size_t SyncFCDsWorkload::size() const { return 0; }

char* SyncFCDsWorkload::marshal() const { return workload::marshal(); }

ace::workload_types SyncFCDsWorkload::get_type() const {
  return ace::workload_types::no_redist;
}

// ---------------------------------------------------------------------------
// DropOldWorkload
// ---------------------------------------------------------------------------
DropOldWorkload::DropOldWorkload()
    : workload(), threshold_epoch_s(-std::numeric_limits<double>::infinity()) {}

DropOldWorkload::DropOldWorkload(double threshold_epoch_s)
    : workload(), threshold_epoch_s(threshold_epoch_s) {}

DropOldWorkload::DropOldWorkload(const std::shared_ptr<char> data, size_t size)
    : workload(data, size) {
  std::memcpy(&threshold_epoch_s, data.get(), sizeof(double));
}

void DropOldWorkload::execute() {
  if (threshold_epoch_s > -std::numeric_limits<double>::infinity()) {
    AdvanceState::instance().gv().drop_old(threshold_epoch_s);
  }
}

size_t DropOldWorkload::size() const { return sizeof(double); }

char* DropOldWorkload::marshal() const {
  char* data = workload::marshal();
  std::memcpy(data, &threshold_epoch_s, sizeof(double));
  return data;
}

ace::workload_types DropOldWorkload::get_type() const {
  return ace::workload_types::no_redist;
}

}  // namespace ruthlib
