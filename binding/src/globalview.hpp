#pragma once

#include <unordered_map>
#include <vector>
#include <shared_mutex>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <set>
#include <mutex>

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


/**
 * SegmentGlobalView - Thread-safe container for FCD records on a single segment
 */
class SegmentGlobalView {
private:
    std::vector<FCDRecord> fcds_;
    mutable std::shared_mutex lock_;

public:
    SegmentGlobalView() = default;

    // Add an FCD record to this segment (write-locked)
    void add(const FCDRecord& fcd) {
        std::unique_lock lock(lock_);
        fcds_.push_back(fcd);
    }

    /* Count vehicles ahead of a given position within a time window */
    int count_vehicles_ahead(double datetime_seconds, double tolerance_seconds, double vehicle_offset_m, int exclude_vehicle_id) const {
        std::shared_lock lock(lock_);

        const double dt_min = datetime_seconds - tolerance_seconds;
        const double dt_max = datetime_seconds + tolerance_seconds;


        std::set<int> unique_vehicles;

        // Iterate through FCDs - compiler can vectorize this loop
        // because FCDs are contiguous in memory and comparisons are independent
        const size_t size = fcds_.size();
        const FCDRecord* data = fcds_.data();
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
        return static_cast<int>(unique_vehicles.size());
    }

    /* Get all FCD records for this segment (read-locked) */
    std::vector<FCDRecord> get_fcds() const {
        std::shared_lock lock(lock_);
        return fcds_;
    }

    /**
     * Remove old FCD records before a given datetime
     * Returns true if any were removed (used to track modifications)
     */
    bool drop_old(double dt_threshold) {
        std::unique_lock lock(lock_);

        const size_t old_size = fcds_.size();

        fcds_.erase(
            std::remove_if(fcds_.begin(), fcds_.end(),
                [dt_threshold](const FCDRecord& fcd) {
                    return fcd.datetime_seconds < dt_threshold;
                }),
            fcds_.end()
        );

        return fcds_.size() != old_size;
    }

    /* Clear all FCD records */
    void clear() {
        std::unique_lock lock(lock_);
        fcds_.clear();
    }

    /* Get count of FCD records */
    size_t size() const {
        std::shared_lock lock(lock_);
        return fcds_.size();
    }
};

/**
 * GlobalView - C++ implementation of ruth/globalview.py
 */
class GlobalView {
private:
    // Hash map: segment_id -> segment's FCD records
    std::unordered_map<int64_t, SegmentGlobalView> segments_;
    mutable std::shared_mutex segments_lock_;


    // Level of Service constants (from Python globalview.py)
    static constexpr double MILE_TO_METERS = 1609.344;
    static constexpr double ENDING_LENGTH = 200.0;

public:
    // LoS ranges: (low_speed, high_speed) -> (m, n) where LoS = m + (density - low) * 0.2 / (high - low)
    struct LoSRange {
        double low_speed;
        double high_speed;
        double m;
        double n;
    };

    static const LoSRange LoS_RANGES[5];

    GlobalView() = default;

    /**
     * Add an FCD record to the global view
     */
    void add(const FCDRecord& fcd) {
        std::unique_lock lock(segments_lock_);
        segments_[fcd.segment_id].add(fcd);
    }

    /**
     * Add multiple FCD records in batch (more efficient than individual adds)
     * Reduces Python-C++ boundary overhead by ~3x
     */
    void add_batch(const std::vector<FCDRecord>& fcds) {
        if (fcds.empty()) return;

        // Group FCDs by segment to minimize lock contention
        std::unordered_map<int64_t, std::vector<FCDRecord>> by_segment;
        for (const auto& fcd : fcds) {
            by_segment[fcd.segment_id].push_back(fcd);
        }

        // Add all FCDs for each segment with a single lock
        for (auto& [segment_id, segment_fcds] : by_segment) {
            std::unique_lock lock(segments_lock_);
            for (const auto& fcd : segment_fcds) {
                segments_[segment_id].add(fcd);
            }
        }
    }

    /**
     * Count vehicles ahead of a given vehicle on a segment
     * Returns unique vehicle count within time window
     */
    int number_of_vehicles_ahead(
        double datetime_seconds,
        int64_t segment_id,
        double tolerance_seconds = 0.0,
        int vehicle_id = -1,
        double vehicle_offset_m = 0.0
    ) const {
        std::shared_lock lock(segments_lock_);

        const auto it = segments_.find(segment_id);
        if (it == segments_.end()) {
            return 0;  // No FCD records for this segment
        }

        return it->second.count_vehicles_ahead(
            datetime_seconds, tolerance_seconds, vehicle_offset_m, vehicle_id
        );
    }

    /**
     * Calculate level of service in front of a vehicle
     * Based on vehicle density and segment parameters
     */
    double level_of_service_in_front_of_vehicle(
        double datetime_seconds,
        int64_t segment_id,
        double segment_length,
        int segment_lanes,
        double tolerance_seconds = 0.0,
        int vehicle_id = -1,
        double vehicle_offset_m = 0.0
    ) const {
        // Count vehicles ahead
        const int n_vehicles = number_of_vehicles_ahead(
            datetime_seconds, segment_id, tolerance_seconds, vehicle_id, vehicle_offset_m
        );

        // Calculate rest of segment length
        const double rest_segment_length = segment_length - vehicle_offset_m;

        // Rescale density: use ENDING_LENGTH to avoid massive LoS increase at segment end
        const double denominator = (rest_segment_length < ENDING_LENGTH) ? ENDING_LENGTH : rest_segment_length;
        const double n_vehicles_per_mile =
            static_cast<double>(n_vehicles) * MILE_TO_METERS / (denominator * segment_lanes);

        // Find LoS by density range
        double los = std::numeric_limits<double>::infinity();
        for (const auto& range : LoS_RANGES) {
            if (n_vehicles_per_mile < range.high_speed) {
                const double d = range.high_speed - range.low_speed;
                los = range.m + ((n_vehicles_per_mile - range.low_speed) * 0.2 / d);
                break;
            }
        }

        // Reverse the level of service (1.0 = 100% LoS, but ranges are inverted)
        return (los == std::numeric_limits<double>::infinity()) ? los : 1.0 - los;
    }

    /**
     * Calculate level of service for entire segment (no vehicle exclusion)
     */
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

    /**
     * Get average speed in m/s on a segment from FCD records
     * Only considers unique vehicles (one speed per vehicle)
     */
    double get_segment_speed(int64_t segment_id) const {
        std::shared_lock lock(segments_lock_);

        const auto it = segments_.find(segment_id);
        if (it == segments_.end()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const auto fcds = it->second.get_fcds();
        if (fcds.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Track speeds by vehicle_id to get unique speeds
        std::unordered_map<int, double> speeds;
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

    /**
     * Remove FCD records older than a given threshold
     * Returns set of segments that were modified
     */
    std::set<int64_t> drop_old(double dt_threshold) {
        std::unique_lock lock(segments_lock_);

        std::set<int64_t> modified;
        for (auto& [segment_id, seg_view] : segments_) {
            if (seg_view.drop_old(dt_threshold)) {
                modified.insert(segment_id);
            }
        }

        return modified;
    }

    /**
     * Clear all FCD records (for testing/reset)
     */
    void clear() {
        std::unique_lock lock(segments_lock_);
        segments_.clear();
    }

    /**
     * Export all FCD records for serialization (pickling)
     * Returns a flat vector of all FCDs across all segments
     */
    std::vector<FCDRecord> export_all_fcds() const {
        std::shared_lock lock(segments_lock_);

        std::vector<FCDRecord> all_fcds;
        for (const auto& [segment_id, seg_view] : segments_) {
            auto fcds = seg_view.get_fcds();
            all_fcds.insert(all_fcds.end(), fcds.begin(), fcds.end());
        }

        return all_fcds;
    }

    /**
     * Import FCD records from serialization (unpickling)
     * Rebuilds internal state from a flat vector of FCDs
     */
    void import_all_fcds(const std::vector<FCDRecord>& fcds) {
        std::unique_lock lock(segments_lock_);

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
