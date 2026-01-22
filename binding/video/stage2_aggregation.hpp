#pragma once

#include "common_types.hpp"
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ==================== STAGE 2: TIME SEGMENT AGGREGATION ====================
// This stage groups vehicle data by time segments and road segments

// Key for grouping by time frame and road segment
struct TimeSegmentKey {
    FrameID frame_id;
    int64_t node_from;
    int64_t node_to;

    bool operator==(const TimeSegmentKey &other) const {
        return frame_id == other.frame_id &&
               node_from == other.node_from &&
               node_to == other.node_to;
    }
};

inline void hash_combine(std::size_t &seed, std::size_t value) noexcept {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <>
struct std::hash<TimeSegmentKey> {
    size_t operator()(const TimeSegmentKey &k) const noexcept {
        size_t seed = 54;
        hash_combine(seed, std::hash<int64_t>()(k.frame_id));
        hash_combine(seed, std::hash<int64_t>()(k.node_from));
        hash_combine(seed, std::hash<int64_t>()(k.node_to));
        return seed;
    }
};

// Aggregated statistics for a time segment
struct AggregatedTimeSegmentData {
    int vehicles_finished = 0;

    double speed_sum_first = 0.0;
    int speed_count_first = 0;
    int count_unique_first = 0;

    double speed_sum_second = 0.0;
    int speed_count_second = 0;
    int count_unique_second = 0;

    void add_speed(double speed, bool pos) {
        if (pos) {
            speed_sum_first += speed;
            speed_count_first++;
        } else {
            speed_sum_second += speed;
            speed_count_second++;
        }
    }

    void increment_vehicles_finished() {
        vehicles_finished++;
    }

    void increment_count_unique(bool pos) {
        if (pos) {
            count_unique_first++;
        } else {
            count_unique_second++;
        }
    }

    [[nodiscard]] double get_avg_speed(bool pos) const {
        if (pos) {
            return speed_count_first > 0 ? speed_sum_first / speed_count_first : 0.0;
        } else {
            return speed_count_second > 0 ? speed_sum_second / speed_count_second : 0.0;
        }
    }

    [[nodiscard]] double get_volume(bool pos) const {
        return pos ? count_unique_first : count_unique_second;
    }
};

// Additional per-frame statistics
struct AddedStats {
    int64_t time = 0;
    float distance = 0.0f;
};

using GroupedMap = std::unordered_map<TimeSegmentKey, AggregatedTimeSegmentData>;
using SeenMap = std::unordered_map<FrameID, std::unordered_set<VehicleID>>;

// Round timestamps to frame IDs and create processing chunks
inline int64_t round_and_chunk(std::vector<VehicleData> &data, int number_of_frames, int round_freq_s,
                               std::vector<std::pair<size_t, size_t>> &ranges, int64_t explicit_round_interval = -1) {
    // Find min/max timestamps (data may not be sorted yet)
    auto [min_it, max_it] = std::minmax_element(data.begin(), data.end(),
        [](const VehicleData &a, const VehicleData &b) { return a.timestamp < b.timestamp; });
    const Timestamp first_ts = min_it->timestamp;
    const Timestamp last_ts = max_it->timestamp;
    int64_t duration = last_ts - first_ts;
    int64_t round_interval = 0;

    if (explicit_round_interval > 0) {
        round_interval = explicit_round_interval;
        if (round_interval < round_freq_s) {
            std::cerr << "Warning: explicit round_interval (" << round_interval
                      << "s) is below file minimum round_freq_s (" << round_freq_s
                      << "s). Using " << round_freq_s << "s instead.\n";
            round_interval = round_freq_s;
        }
    } else {
        if (number_of_frames <= 0)
            throw std::runtime_error("Invalid number_of_frames (0) when explicit round interval not provided.");
        round_interval = duration / number_of_frames;
        if (round_interval < round_freq_s) round_interval = round_freq_s;
    }

    for (auto &d : data) {
        d.frame_id = ((d.timestamp + round_interval / 2) / round_interval);
    }

    // Sort by timestamp (necessary due to parallel aggregation later)
    std::cout << "Sorting by timestamp...\n";
    std::stable_sort(data.begin(), data.end(), [](const VehicleData &a, const VehicleData &b) {
        return a.timestamp < b.timestamp;
    });
    std::cout << "Sorting complete\n";

#ifdef DEBUG
    // Check if timestamps are already ordered per vehicle (after sorting by timestamp)
    std::cout << "Checking timestamp ordering per vehicle...\n";
    std::unordered_map<VehicleID, std::pair<Timestamp, size_t>> last_seen_ts_idx;
    size_t total_violations = 0;
    size_t vehicles_with_violations = 0;
    std::unordered_set<VehicleID> vehicles_violated;
    size_t duplicate_ts_count = 0;
    size_t duplicate_ts_shown = 0;

    for (size_t i = 0; i < data.size(); ++i) {
        const auto &d = data[i];
        auto it = last_seen_ts_idx.find(d.vehicle_id);
        if (it != last_seen_ts_idx.end()) {
            auto [last_ts, last_idx] = it->second;

            // Check for timestamp going backwards
            if (d.timestamp < last_ts) {
                total_violations++;
                if (vehicles_violated.insert(d.vehicle_id).second) {
                    vehicles_with_violations++;
                    if (vehicles_with_violations <= 5) {
                        std::cout << "  ⚠ Vehicle " << d.vehicle_id
                                  << ": timestamp went backwards (" << last_ts
                                  << " -> " << d.timestamp << ")\n";
                    }
                }
            }
            // Check for duplicate timestamp (same vehicle, same timestamp)
            else if (d.timestamp == last_ts) {
                duplicate_ts_count++;
                if (duplicate_ts_shown < 5) {
                    const auto &prev = data[last_idx];
                    std::cout << "  🔄 Vehicle " << d.vehicle_id << " has duplicate timestamp " << d.timestamp << ":\n";
                    std::cout << "      Record[" << last_idx << "]: seg=(" << prev.node_from << "->" << prev.node_to
                              << "), offset=" << prev.start_offset_m << ", speed=" << prev.speed_mps << "\n";
                    std::cout << "      Record[" << i << "]: seg=(" << d.node_from << "->" << d.node_to
                              << "), offset=" << d.start_offset_m << ", speed=" << d.speed_mps << "\n";
                    duplicate_ts_shown++;
                }
            }
        }
        last_seen_ts_idx[d.vehicle_id] = {d.timestamp, i};
    }

    if (duplicate_ts_count > 0) {
        std::cout << "  ℹ️  Found " << duplicate_ts_count << " records with duplicate timestamps (same vehicle, same ts)\n";
        if (duplicate_ts_shown < duplicate_ts_count) {
            std::cout << "      (showing first " << duplicate_ts_shown << " examples)\n";
        }
    }

    if (total_violations == 0) {
        std::cout << "  ✅ All timestamps are monotonically increasing per vehicle\n";
        std::cout << "  → Data is in correct order after timestamp sort\n";
    } else {
        std::cout << "  ❌ Found " << total_violations << " timestamp violations across "
                  << vehicles_with_violations << " vehicles\n";
        std::cout << "  → WARNING: This should not happen after sorting by timestamp!\n";
    }
#endif

    size_t begin = 0;
    size_t chunk_size = 1'000'000;
    for (size_t i = 1; i < data.size(); i++) {
        if (data[i].frame_id != data[i-1].frame_id && (i - begin) > chunk_size) {
            ranges.emplace_back(begin, i);
            begin = i;
        }
    }
    ranges.emplace_back(begin, data.size());

    return round_interval;
}

// Get max vehicle ID from data
inline VehicleID get_max_vehicle_id(const std::vector<VehicleData> &data) {
    VehicleID vehicle_id_max = 0;
    std::unordered_set<int64_t> unique_vehicles;
    unique_vehicles.reserve(data.size() / 100);

    for (const auto &d : data) {
        unique_vehicles.insert(d.vehicle_id);
        vehicle_id_max = std::max(vehicle_id_max, d.vehicle_id);
    }
    std::cout << "\t Unique vehicles in data: " << unique_vehicles.size() << "\n";
    std::cout << "\t Max vehicle ID: " << vehicle_id_max << "\n";

    return vehicle_id_max;
}

// Aggregate a single chunk (used in streaming mode)
inline void aggregate_chunk(const std::vector<VehicleData> &chunk,
                            GroupedMap &grouped,
                            SeenMap &seen,
                            VehicleID &max_vehicle_id,
                            Timestamp &min_ts,
                            Timestamp &max_ts) {
    for (const auto &d : chunk) {
        TimeSegmentKey key{d.frame_id, d.node_from, d.node_to};

        auto &agg = grouped[key];
        bool pos = (d.segment_length > 0) ? (d.start_offset_m / static_cast<float>(d.segment_length)) < 0.5f : true;
        agg.add_speed(d.speed_mps, pos);

        // Track unique vehicles per frame (vehicle counted once per frame, any record)
        if (seen[d.frame_id].insert(d.vehicle_id).second) {
            agg.increment_count_unique(pos);
        }

        if (!d.active)
            agg.increment_vehicles_finished();

        max_vehicle_id = std::max(max_vehicle_id, d.vehicle_id);
        min_ts = std::min(min_ts, d.timestamp);
        max_ts = std::max(max_ts, d.timestamp);
    }
}


// Parallel aggregation using OpenMP (legacy mode)
inline void aggregate_parallel(const std::vector<VehicleData> &data,
                               const std::vector<std::pair<size_t, size_t>> &ranges,
                               std::vector<GroupedMap> &grouped_local,
                               std::vector<SeenMap> &seen_local) {
#pragma omp parallel for schedule(dynamic) default(none) shared(data, ranges, grouped_local, seen_local)
    for (const auto& [begin, end] : ranges) {
        int thread_id = omp_get_thread_num();
        for (size_t i = begin; i < end; ++i) {
            const auto &d = data[i];
            int64_t frame_id = d.frame_id;
            TimeSegmentKey key{frame_id, d.node_from, d.node_to};

            auto &agg = grouped_local[thread_id][key];
            bool pos = (d.segment_length > 0) ? (d.start_offset_m / static_cast<float>(d.segment_length)) < 0.5f : true;
            agg.add_speed(d.speed_mps, pos);

            if (seen_local[thread_id][frame_id].insert(d.vehicle_id).second)
                agg.increment_count_unique(pos);

            if (!d.active)
                agg.increment_vehicles_finished();
        }
    }
}

// Compute added stats (distance/time) from sorted data
// Expects data sorted oldest→newest (ascending timestamps)
inline std::unordered_map<FrameID, AddedStats> compute_added_stats(const std::vector<VehicleData> &data, int vehicles_count) {
    std::vector<int> previous_vehicle_info(vehicles_count + 1, -1);
    std::unordered_map<FrameID, AddedStats> added_stats{};

    const auto earliest_timestamp = data.front().timestamp;  // front() = oldest

    // Process oldest→newest, tracking previous record for each vehicle
    for (size_t i = 0; i < data.size(); ++i) {
        const auto &fcd = data[i];
        auto vehicle_idx = fcd.vehicle_id;

        if (previous_vehicle_info[vehicle_idx] != -1) {
            // We've seen this vehicle before - compute delta from previous to current
            int prev_idx = previous_vehicle_info[vehicle_idx];
            const auto &prev = data[prev_idx];
            auto delta_time = fcd.timestamp - prev.timestamp;
            if (delta_time < 0){
                            std::cout << "Delta time is " << delta_time << " for vehicle " << vehicle_idx
                          << " (prev ts: " << prev.timestamp << ", curr ts: " << fcd.timestamp << ")\n";
                //throw std::runtime_error("Timestamps not sorted correctly!");
            }
            added_stats[fcd.frame_id].time += delta_time;
            if (fcd.node_from == prev.node_from && fcd.node_to == prev.node_to)
                added_stats[fcd.frame_id].distance += fcd.start_offset_m - prev.start_offset_m;
            else
                added_stats[fcd.frame_id].distance += fcd.start_offset_m;
        } else {
            // First record for this vehicle - delta from simulation start
            auto delta_time = fcd.timestamp - earliest_timestamp;
            if (delta_time < 0){
              std::cout << "Delta time is " << delta_time << " for vehicle " << vehicle_idx
                        << " (earliest ts: " << earliest_timestamp << ", curr ts: " << fcd.timestamp << ")\n";
                //throw std::runtime_error("Timestamps not sorted correctly!");
            }
            added_stats[fcd.frame_id].time += delta_time;
            added_stats[fcd.frame_id].distance += fcd.start_offset_m;
        }
        previous_vehicle_info[vehicle_idx] = static_cast<int>(i);
    }

    return added_stats;
}

// Merge thread-local results into global structures
inline void merge_thread_results(std::vector<GroupedMap> &grouped_local,
                                 std::vector<SeenMap> &seen_local,
                                 GroupedMap &grouped,
                                 int &max_unique_vehicle_count,
                                 int &max_unique_on_segment_count,
                                 std::unordered_map<FrameID, int64_t> &total_seen) {
    size_t nthreads = grouped_local.size();
    grouped.reserve(std::accumulate(grouped_local.begin(), grouped_local.end(), 0ul,
                                    [](size_t s, const auto &m) { return s + m.size(); }));

    for (size_t t = 0; t < nthreads; ++t) {
        for (auto &[key, agg] : grouped_local[t]) {
            if (auto it = grouped.find(key); it != grouped.end())
                throw std::runtime_error("Key collision during merge!");
            grouped.emplace(key, agg);

            auto unique_count = static_cast<int>(seen_local[t][key.frame_id].size());
            max_unique_vehicle_count = std::max(max_unique_vehicle_count, unique_count);
            max_unique_on_segment_count = std::max({
                max_unique_on_segment_count,
                agg.count_unique_first,
                agg.count_unique_second
            });
        }
    }

    std::cout << "\t Max unique vehicles in a time segment: " << max_unique_vehicle_count << "\n";
    std::cout << "\t Max unique vehicles on a segment: " << max_unique_on_segment_count << "\n";

    // Calculates stat for active vehicles in frame
    for (size_t t = 0; t < nthreads; ++t)
        for (const auto &[frame_id, ids] : seen_local[t])
            total_seen[frame_id] += static_cast<int64_t>(ids.size());
}

// Result of streaming aggregation
struct StreamingAggregationResult {
    GroupedMap grouped;
    std::unordered_map<FrameID, int64_t> total_seen;
    std::unordered_map<FrameID, AddedStats> added_stats;
    VehicleID max_vehicle_id = 0;
    int max_unique_vehicle_count = 0;
    int max_unique_on_segment_count = 0;
    Timestamp actual_min_ts = std::numeric_limits<Timestamp>::max();
    Timestamp actual_max_ts = std::numeric_limits<Timestamp>::min();
};
