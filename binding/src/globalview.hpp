#pragma once

#include <unordered_map>
#include <vector>
#include <deque>
#include <shared_mutex>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <set>
#include <mutex>
#include <unordered_set>

struct FCDRecord {
    double datetime_seconds;      // Timestamp as seconds (epoch)
    int vehicle_id;
    int64_t segment_id;
    double offset_from_start;     // Position on segment (meters)
    double vehicle_speed_mps;     // Speed in meters per second

    FCDRecord() = default;

    FCDRecord(double dt, int vid, int64_t sid, double offset, double speed)
        : datetime_seconds(dt), vehicle_id(vid), segment_id(sid),
          offset_from_start(offset), vehicle_speed_mps(speed) {}
};

struct FCDBatch {
    std::vector<FCDRecord> fcds;
    double max_datetime_seconds;

    FCDBatch() : max_datetime_seconds(0.0) {}

    explicit FCDBatch(std::vector<FCDRecord>&& records, double max_dt)
        : fcds(std::move(records)), max_datetime_seconds(max_dt) {}
};

class SegmentGlobalView {
private:
    std::deque<FCDBatch> fcd_batches_;
public:
    SegmentGlobalView() = default;

    void add(const FCDRecord& fcd) {
        std::vector<FCDRecord> single_vec;
        single_vec.reserve(1);
        single_vec.push_back(fcd);
        fcd_batches_.emplace_back(std::move(single_vec), fcd.datetime_seconds);
    }

    void add_batch(const std::vector<FCDRecord>& fcds) {
        if (fcds.empty()) return;

        double max_dt = fcds[0].datetime_seconds;
        for (size_t i = 1; i < fcds.size(); ++i) {
            if (fcds[i].datetime_seconds > max_dt) {
                max_dt = fcds[i].datetime_seconds;
            }
        }
        fcd_batches_.emplace_back(std::vector<FCDRecord>(fcds), max_dt);
    }

    void add_batch(std::vector<FCDRecord>&& fcds) {
        if (fcds.empty()) return;

        double max_dt = fcds[0].datetime_seconds;
        for (size_t i = 1; i < fcds.size(); ++i) {
            if (fcds[i].datetime_seconds > max_dt) {
                max_dt = fcds[i].datetime_seconds;
            }
        }
        fcd_batches_.emplace_back(std::move(fcds), max_dt);
    }

    int count_vehicles_ahead(double datetime_seconds, double tolerance_seconds, double vehicle_offset_m, int exclude_vehicle_id) const {
        const double dt_min = datetime_seconds - tolerance_seconds;
        const double dt_max = datetime_seconds + tolerance_seconds;

        std::unordered_set<int> unique_vehicles;
        unique_vehicles.reserve(64);  // Pre-allocate reasonable size

        for (const auto& batch : fcd_batches_) {
            // Skip entire batch if it's too old
            if (batch.max_datetime_seconds < dt_min) {
                continue;
            }

            const size_t size = batch.fcds.size();
            const FCDRecord* data = batch.fcds.data();

            for (size_t i = 0; i < size; ++i) {
                const FCDRecord& fcd = data[i];
                if (fcd.offset_from_start <= vehicle_offset_m) {
                    continue;
                }
                if (fcd.vehicle_id == exclude_vehicle_id) {
                    continue;
                }
                if (fcd.datetime_seconds >= dt_min && fcd.datetime_seconds <= dt_max) {
                    unique_vehicles.insert(fcd.vehicle_id);
                }
            }
        }
        return static_cast<int>(unique_vehicles.size());
    }

    std::vector<FCDRecord> get_fcds() const {
        std::vector<FCDRecord> result;
        size_t total_size = 0;
        for (const auto& batch : fcd_batches_) {
            total_size += batch.fcds.size();
        }
        result.reserve(total_size);
        for (const auto& batch : fcd_batches_) {
            result.insert(result.end(), batch.fcds.begin(), batch.fcds.end());
        }
        return result;
    }

    bool drop_old(double dt_threshold) {
        bool modified = false;
        while (!fcd_batches_.empty() && fcd_batches_.front().max_datetime_seconds < dt_threshold) {
            fcd_batches_.pop_front();
            modified = true;
        }
        return modified;
    }

    void clear() {
        fcd_batches_.clear();
    }

    size_t size() const {
        size_t total = 0;
        for (const auto& batch : fcd_batches_) {
            total += batch.fcds.size();
        }
        return total;
    }
};

class GlobalView {
private:
    // Hash map: segment_id -> segment's FCD records
    std::unordered_map<int64_t, SegmentGlobalView> segments_;

    static constexpr double MILE_TO_METERS = 1609.344;
    static constexpr double ENDING_LENGTH = 200.0;

public:
    struct LoSRange {
        double low_speed;
        double high_speed;
        double m;
        double n;
    };

    static const LoSRange LoS_RANGES[5];

    GlobalView() = default;

    void add(const FCDRecord& fcd) {
        segments_[fcd.segment_id].add(fcd);
    }

    void add_batch(const std::vector<FCDRecord>& fcds) {
        if (fcds.empty()) return;
        std::unordered_map<int64_t, std::vector<FCDRecord>> by_segment;
        by_segment.reserve(32);  // Pre-allocate for common case

        for (const auto& fcd : fcds) {
            by_segment[fcd.segment_id].push_back(fcd);
        }

        for (auto& [segment_id, segment_fcds] : by_segment) {
            segments_[segment_id].add_batch(segment_fcds);
        }
    }

    int number_of_vehicles_ahead(
        double datetime_seconds,
        int64_t segment_id,
        double tolerance_seconds = 0.0,
        int vehicle_id = -1,
        double vehicle_offset_m = 0.0
    ) const {
        const auto it = segments_.find(segment_id);
        if (it == segments_.end()) {
            return 0;  // No FCD records for this segment
        }

        return it->second.count_vehicles_ahead(
            datetime_seconds, tolerance_seconds, vehicle_offset_m, vehicle_id
        );
    }

    double level_of_service_in_front_of_vehicle(
        double datetime_seconds,
        int64_t segment_id,
        double segment_length,
        int segment_lanes,
        double tolerance_seconds = 0.0,
        int vehicle_id = -1,
        double vehicle_offset_m = 0.0
    ) const {
        const int n_vehicles = number_of_vehicles_ahead(
            datetime_seconds, segment_id, tolerance_seconds, vehicle_id, vehicle_offset_m
        );

        const double rest_segment_length = segment_length - vehicle_offset_m;

        const double denominator = (rest_segment_length < ENDING_LENGTH) ? ENDING_LENGTH : rest_segment_length;
        const double n_vehicles_per_mile =
            static_cast<double>(n_vehicles) * MILE_TO_METERS / (denominator * segment_lanes);

        double los = std::numeric_limits<double>::infinity();
        for (const auto& range : LoS_RANGES) {
            if (n_vehicles_per_mile < range.high_speed) {
                const double d = range.high_speed - range.low_speed;
                los = range.m + ((n_vehicles_per_mile - range.low_speed) * 0.2 / d);
                break;
            }
        }

        return (los == std::numeric_limits<double>::infinity()) ? los : 1.0 - los;
    }

    double level_of_service_in_time_at_segment(
        double datetime_seconds,
        int64_t segment_id,
        double segment_length,
        int segment_lanes,
        double tolerance_seconds = 0.0
    ) const {
        return level_of_service_in_front_of_vehicle(
            datetime_seconds, segment_id, segment_length, segment_lanes,
            tolerance_seconds, -1, 0.0
        );
    }

    double get_segment_speed(int64_t segment_id) const {
        const auto it = segments_.find(segment_id);
        if (it == segments_.end()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const auto fcds = it->second.get_fcds();
        if (fcds.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        std::unordered_map<int, double> speeds;
        speeds.reserve(fcds.size() / 2);  // Pre-allocate

        for (const auto& fcd : fcds) {
            speeds[fcd.vehicle_id] = fcd.vehicle_speed_mps;
        }

        // Calculate average
        double sum = 0.0;
        for (const auto& [vid, speed] : speeds) {
            sum += speed;
        }

        return sum / static_cast<double>(speeds.size());
    }

    std::set<int64_t> drop_old(double dt_threshold) {
        std::vector<int64_t> modified_vec;

        std::vector<std::pair<int64_t, SegmentGlobalView*>> segment_ptrs;
        segment_ptrs.reserve(segments_.size());
        for (auto& [segment_id, seg_view] : segments_) {
            segment_ptrs.emplace_back(segment_id, &seg_view);
        }

        #pragma omp parallel
        {
            std::vector<int64_t> local_modified;

            #pragma omp for nowait
            for (size_t i = 0; i < segment_ptrs.size(); ++i) {
                if (segment_ptrs[i].second->drop_old(dt_threshold)) {
                    local_modified.push_back(segment_ptrs[i].first);
                }
            }

            #pragma omp critical
            {
                modified_vec.insert(modified_vec.end(), local_modified.begin(), local_modified.end());
            }
        }

        return std::set<int64_t>(modified_vec.begin(), modified_vec.end());
    }

    void clear() {
        segments_.clear();
    }

    std::vector<FCDRecord> export_all_fcds() const {
        std::vector<FCDRecord> all_fcds;
        for (const auto& [segment_id, seg_view] : segments_) {
            auto fcds = seg_view.get_fcds();
            all_fcds.insert(all_fcds.end(), fcds.begin(), fcds.end());
        }

        return all_fcds;
    }

    void import_all_fcds(const std::vector<FCDRecord>& fcds) {
        segments_.clear();

        for (const auto& fcd : fcds) {
            segments_[fcd.segment_id].add(fcd);
        }
    }
};

// Initialize static LoS ranges
// (low_speed_kph, high_speed_kph) -> (m, n) coefficients
inline const GlobalView::LoSRange GlobalView::LoS_RANGES[5] = {
    {0, 12, 0.0, 0.2},
    {12, 20, 0.2, 0.4},
    {20, 30, 0.4, 0.6},
    {30, 42, 0.6, 0.8},
    {42, 67, 0.8, 1.0}
};
