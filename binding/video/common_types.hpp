#pragma once

#include <cstdint>
#include <limits>
#include <string>

// ==================== COMMON TYPES USED ACROSS ALL STAGES ====================

// Basic type aliases
using Timestamp = int64_t;
using FrameID = int64_t;
using VehicleID = int64_t;

// Configuration for the entire pipeline
struct Config {
    std::string filename;
    std::string dataset_name = "fcd";
    std::string outfile = "fcd_aggregated.h5";
    int length_s = -1;
    int fps = 25;
    size_t max_records = std::numeric_limits<size_t>::max();
    int round_interval_s = -1;
    bool streaming_mode = false;
    uint64_t chunk_size = 500'000;
};

// Raw vehicle data from HDF5 (used in Stage 1 loading)
struct VehicleData {
    FrameID frame_id = 0;
    Timestamp timestamp = 0;
    int64_t node_from = 0;
    int64_t node_to = 0;
    int32_t segment_length = 0;
    VehicleID vehicle_id = 0;
    float start_offset_m = 0.0f;
    float speed_mps = 0.0f;
    bool active = false;
};
