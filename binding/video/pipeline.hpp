#pragma once

#include "stage1_loading.hpp"
#include "stage2_aggregation.hpp"
#include "stage3_output.hpp"
#include <iostream>

// ==================== COMPLETE STREAMING PIPELINE ====================

// Stream and aggregate multiple files with chunked reading
inline StreamingAggregationResult stream_aggregate_files(const std::vector<std::string> &files, const std::string &dataset_name,
                                                          int64_t round_interval, uint64_t chunk_size) {
    StreamingAggregationResult result;
    SeenMap global_seen;

    std::cout << "Streaming aggregation with chunk size: " << chunk_size << "\n";
    size_t total_records_processed = 0;

    for (const auto &filepath : files) {
        std::cout << "  Processing: " << filepath << "\n";
        ChunkedHDF5Reader reader(filepath, dataset_name, chunk_size);

        while (!reader.is_exhausted()) {
            auto chunk = reader.read_next_chunk();
            if (chunk.empty()) break;

            total_records_processed += chunk.size();

            // Assign frame IDs based on round interval
            for (auto &d : chunk) {
                d.frame_id = (d.timestamp + round_interval / 2) / round_interval;
            }

            // Aggregate this chunk (simple aggregation, no added_stats)
            aggregate_chunk(chunk, result.grouped, global_seen,
                          result.max_vehicle_id, result.actual_min_ts, result.actual_max_ts);
        }
    }

    std::cout << "  Total records processed: " << total_records_processed << "\n";
    std::cout << "  Unique (frame, segment) keys: " << result.grouped.size() << "\n";

    // Compute total_seen and max counts from global_seen
    for (const auto &[frame_id, ids] : global_seen) {
        result.total_seen[frame_id] = static_cast<int64_t>(ids.size());
        result.max_unique_vehicle_count = std::max(result.max_unique_vehicle_count,
                                                    static_cast<int>(ids.size()));
    }

    for (const auto &[key, agg] : result.grouped) {
        result.max_unique_on_segment_count = std::max({
            result.max_unique_on_segment_count,
            agg.count_unique_first,
            agg.count_unique_second
        });
    }

    std::cout << "  Max vehicle ID: " << result.max_vehicle_id << "\n";
    std::cout << "  Max unique vehicles in a frame: " << result.max_unique_vehicle_count << "\n";
    std::cout << "  Max unique vehicles on segment: " << result.max_unique_on_segment_count << "\n";

    // added_stats remains empty for streaming mode

    return result;
}
