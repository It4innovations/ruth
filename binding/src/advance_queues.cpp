#include "advance_queues.hpp"

#include <cstdio>
#include <cstdlib>

#include "comm/comm_interface.hpp"

namespace ruthlib {

namespace {

// Same reasoning as advance_store.cpp: one rank calling std::abort() strands the
// others in a collective, so tear the job down through ACE when it is up.
[[noreturn]] void fatal() {
  std::fflush(stderr);
  if (auto comm = ace::CommInterface::instance()) {
    comm->abort();
  }
  std::abort();
}

}  // namespace

QueueRegistry& QueueRegistry::instance() {
  static QueueRegistry inst;
  return inst;
}

void QueueRegistry::clear() {
  index_.clear();
  order_.clear();
  queues_.clear();
  to_be_added_.clear();
  to_be_removed_.clear();
}

size_t QueueRegistry::slot_for(int64_t node_from, int64_t node_to) {
  const Key key{node_from, node_to};
  const auto it = index_.find(key);
  if (it != index_.end()) {
    return it->second;
  }
  const size_t slot = order_.size();
  index_.emplace(key, slot);
  order_.push_back(key);
  queues_.emplace_back();
  return slot;
}

std::deque<int32_t>& QueueRegistry::touch(int64_t node_from, int64_t node_to) {
  return queues_[slot_for(node_from, node_to)];
}

bool QueueRegistry::contains(int64_t node_from, int64_t node_to,
                             int32_t vehicle_id) const {
  const auto it = index_.find(Key{node_from, node_to});
  if (it == index_.end()) {
    return false;
  }
  const std::deque<int32_t>& queue = queues_[it->second];
  for (const int32_t id : queue) {
    if (id == vehicle_id) {
      return true;
    }
  }
  return false;
}

void QueueRegistry::stage_add(int32_t vehicle_id, int64_t node_from, int64_t node_to) {
  to_be_added_.push_back(StagedOp{vehicle_id, node_from, node_to});
}

void QueueRegistry::stage_remove(int32_t vehicle_id, int64_t node_from, int64_t node_to) {
  to_be_removed_.push_back(StagedOp{vehicle_id, node_from, node_to});
}

void QueueRegistry::batch_update() {
  // Removals first, in the order they were staged. Each pops the FRONT: a
  // vehicle only ever leaves a queue by reaching its head, so finding a
  // different id there means the replay order diverged from the queue order,
  // which would silently corrupt every later step. Python asserts both of
  // these; so do we, loudly.
  for (const StagedOp& op : to_be_removed_) {
    std::deque<int32_t>& queue = touch(op.node_from, op.node_to);
    if (queue.empty()) {
      std::fprintf(stderr,
                   "[ruth] QueueRegistry::batch_update: queue (%lld -> %lld) is empty but "
                   "vehicle %d was staged for removal from it.\n",
                   static_cast<long long>(op.node_from),
                   static_cast<long long>(op.node_to), op.vehicle_id);
      fatal();
    }
    const int32_t popped = queue.front();
    queue.pop_front();
    if (popped != op.vehicle_id) {
      std::fprintf(stderr,
                   "[ruth] QueueRegistry::batch_update: queue (%lld -> %lld) had vehicle %d "
                   "at its head but vehicle %d was staged for removal; the replay order and "
                   "the queue order have diverged.\n",
                   static_cast<long long>(op.node_from),
                   static_cast<long long>(op.node_to), popped, op.vehicle_id);
      fatal();
    }
  }
  to_be_removed_.clear();

  for (const StagedOp& op : to_be_added_) {
    touch(op.node_from, op.node_to).push_back(op.vehicle_id);
  }
  to_be_added_.clear();
}

void QueueRegistry::snapshot(std::vector<int64_t>& node_from,
                             std::vector<int64_t>& node_to,
                             std::vector<int32_t>& offsets,
                             std::vector<int32_t>& vehicle_ids) const {
  node_from.clear();
  node_to.clear();
  offsets.clear();
  vehicle_ids.clear();

  node_from.reserve(order_.size());
  node_to.reserve(order_.size());
  offsets.reserve(order_.size() + 1);

  int32_t acc = 0;
  for (size_t slot = 0; slot < order_.size(); ++slot) {
    node_from.push_back(order_[slot].node_from);
    node_to.push_back(order_[slot].node_to);
    offsets.push_back(acc);
    for (const int32_t id : queues_[slot]) {
      vehicle_ids.push_back(id);
    }
    acc += static_cast<int32_t>(queues_[slot].size());
  }
  offsets.push_back(acc);
}

}  // namespace ruthlib
