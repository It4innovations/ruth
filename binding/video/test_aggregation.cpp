#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include "stage2_aggregation.hpp"

// Helper function to create test vehicle data
VehicleData create_vehicle(Timestamp ts, VehicleID vid, int64_t node_from, int64_t node_to,
                           float speed, float offset, int32_t seg_len = 100, bool active = true) {
    return {
        .frame_id = 0,
        .timestamp = ts,
        .node_from = node_from,
        .node_to = node_to,
        .segment_length = seg_len,
        .vehicle_id = vid,
        .start_offset_m = offset,
        .speed_mps = speed,
        .active = active
    };
}

void test_speed_aggregation() {
    std::cout << "TEST: Speed aggregation\n";

    std::vector<VehicleData> data = {
        create_vehicle(1000, 1, 10, 20, 10.0f, 25.0f),
        create_vehicle(1000, 2, 10, 20, 20.0f, 25.0f),
    };

    GroupedMap grouped;
    SeenMap seen;
    VehicleID max_vid = 0;
    Timestamp min_ts = std::numeric_limits<Timestamp>::max();
    Timestamp max_ts = std::numeric_limits<Timestamp>::min();

    for (auto &d : data) {
        d.frame_id = 0;
    }

    aggregate_chunk(data, grouped, seen, max_vid, min_ts, max_ts);

    assert(grouped.size() == 1);
    auto &agg = grouped.begin()->second;
    assert(agg.speed_count_first == 2);
    assert(std::abs(agg.speed_sum_first - 30.0) < 0.01);

    std::cout << "  ✓ PASSED\n";
}

void test_unique_vehicle_counting() {
    std::cout << "TEST: Unique vehicle counting per frame\n";

    // Vehicle 1 appears on 2 segments, Vehicle 2 appears on 1 segment - all in same frame
    std::vector<VehicleData> data = {
        create_vehicle(1000, 1, 10, 20, 10.0f, 50.0f),   // Vehicle 1, Segment A, Frame 0
        create_vehicle(1000, 1, 30, 40, 10.0f, 50.0f),   // Vehicle 1, Segment B, Frame 0 (same vehicle, different segment)
        create_vehicle(1000, 2, 10, 20, 15.0f, 60.0f),   // Vehicle 2, Segment A, Frame 0
    };

    GroupedMap grouped;
    SeenMap seen;
    VehicleID max_vid = 0;
    Timestamp min_ts = std::numeric_limits<Timestamp>::max();
    Timestamp max_ts = std::numeric_limits<Timestamp>::min();

    for (auto &d : data) {
        d.frame_id = 0;  // All in same frame
    }

    aggregate_chunk(data, grouped, seen, max_vid, min_ts, max_ts);

    // Should have 3 keys: (0, 10, 20), (0, 30, 40), (0, 10, 20) - wait, that's only 2 unique segments
    // Actually: (0, 10, 20), (0, 30, 40), and another record on (0, 10, 20)
    // So 2 unique segment keys
    assert(grouped.size() == 2);

    // seen[0] should contain vehicles 1 and 2 (2 unique vehicles in frame 0)
    assert(seen[0].size() == 2);

    // Total unique count across all segments should be 2
    // Vehicle 1 counted once when first seen, Vehicle 2 counted once when first seen
    int total_unique = 0;
    for (const auto &[key, agg] : grouped) {
        total_unique += agg.count_unique_first + agg.count_unique_second;
    }
    assert(total_unique == 2);

    std::cout << "  ✓ PASSED\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Running C++ Aggregation Tests\n";
    std::cout << "========================================\n\n";

    try {
        test_speed_aggregation();
        test_unique_vehicle_counting();

        std::cout << "\n========================================\n";
        std::cout << "✅ All tests PASSED!\n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception &e) {
        std::cout << "\n❌ Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
