#include "advance_state.hpp"

namespace ruthlib {

AdvanceState& AdvanceState::instance() {
  static AdvanceState inst;
  return inst;
}

GlobalView& AdvanceState::gv() { return gv_; }

double AdvanceState::los_tolerance_s() const { return los_tolerance_s_; }
void AdvanceState::set_los_tolerance_s(double t) { los_tolerance_s_ = t; }

void AdvanceState::reset_step() {
  std::lock_guard<std::mutex> g(mtx_);
  step_fcds_.clear();
  step_deltas_.clear();
}

void AdvanceState::append_fcds(const std::vector<FCDRecord>& fcds) {
  if (fcds.empty()) return;  // nothing to add: skip taking the lock
  std::lock_guard<std::mutex> g(mtx_);
  step_fcds_.insert(step_fcds_.end(), fcds.begin(), fcds.end());
}

void AdvanceState::append_delta(const AdvanceDelta& d) {
  std::lock_guard<std::mutex> g(mtx_);
  step_deltas_.push_back(d);
}

std::vector<FCDRecord> AdvanceState::take_local_fcds() {
  std::lock_guard<std::mutex> g(mtx_);
  std::vector<FCDRecord> out;
  out.swap(step_fcds_);  // step_fcds_ is now guaranteed empty for the next step
  return std::move(out);
}

std::vector<AdvanceDelta> AdvanceState::take_local_deltas() {
  std::lock_guard<std::mutex> g(mtx_);
  std::vector<AdvanceDelta> out;
  out.swap(step_deltas_);  // step_deltas_ is now guaranteed empty for the next step
  return std::move(out);
}

void AdvanceState::set_last_synced_fcds(std::vector<FCDRecord>&& f) {
  std::lock_guard<std::mutex> g(mtx_);
  last_synced_fcds_ = std::move(f);
}

void AdvanceState::set_last_synced_deltas(std::vector<AdvanceDelta>&& d) {
  std::lock_guard<std::mutex> g(mtx_);
  last_synced_deltas_ = std::move(d);
}

std::vector<FCDRecord> AdvanceState::take_last_synced_fcds() {
  std::lock_guard<std::mutex> g(mtx_);
  std::vector<FCDRecord> out;
  out.swap(last_synced_fcds_);
  return std::move(out);
}

std::vector<AdvanceDelta> AdvanceState::take_last_synced_deltas() {
  std::lock_guard<std::mutex> g(mtx_);
  std::vector<AdvanceDelta> out;
  out.swap(last_synced_deltas_);
  return std::move(out);
}

}  // namespace ruthlib
