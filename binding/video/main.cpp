/*
    A C++ program that reads simulation output fcd_history.h5,
    processes it to aggregate statistics over time segments and road segments,
    and writes the aggregated results back to a new HDF5 file which is used for video generation.

    The program uses parallel processing with OpenMP to speed up the aggregation step.
*/

#include "pipeline.hpp"

#include <iostream>
#include <chrono>
#include <getopt.h>
#include <filesystem>
#include <algorithm>
#include <omp.h>

/**
 * Parse command-line arguments into a Config struct.
 */
Config parseArgs(int argc, char* argv[]) {
    Config cfg;

    static struct option long_options[] = {
        {"outfile", required_argument, nullptr, 'o'},
        {"length", required_argument, nullptr, 'l'},
        {"round-interval", required_argument, nullptr, 'r'},
        {"fps", required_argument, nullptr, 'p'},
        {"maxrecords", required_argument, nullptr, 'm'},
        {"streaming", no_argument, nullptr, 's'},
        {"chunk-size", required_argument, nullptr, 'c'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "o:l:p:m:r:sc:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 'o': cfg.outfile = optarg; break;
            case 'l': cfg.length_s = std::stoi(optarg); break;
            case 'r': cfg.round_interval_s = std::stoi(optarg); break;
            case 'p': cfg.fps = std::stoi(optarg); break;
            case 'm': cfg.max_records = std::stoull(optarg); break;
            case 's': cfg.streaming_mode = true; break;
            case 'c': cfg.chunk_size = std::stoull(optarg); break;
            case 'h':
                std::cout << "Usage: ./video_preprocess <INPUT_FILEDIR> [OPTIONS]\n\n"
                          << "Options:\n"
                          << "  -o, --outfile FILE       Output file (default: fcd_aggregated.h5)\n"
                          << "  -l, --length SEC         Video length in seconds\n"
                          << "  -r, --round-interval SEC Explicit rounding interval in seconds\n"
                          << "  -p, --fps N              Frames per second (default: 25)\n"
                          << "  -m, --maxrecords N       Maximum records to read\n"
                          << "  -s, --streaming          Enable streaming mode (lower memory)\n"
                          << "  -c, --chunk-size N       Records per chunk in streaming mode\n"
                          << "  -h, --help               Show this help\n";
                exit(0);
            default:
                break;
        }
    }

    if (optind < argc) {
        cfg.filename = argv[optind];
    } else {
        std::cerr << "Error: filename is required\n";
        exit(1);
    }

    if (cfg.length_s <= 0 && cfg.round_interval_s <= 0) {
        std::cerr << "Error: you must specify either --length <seconds> or --round-interval <seconds>\n";
        std::cerr << "Usage: ./video_preprocess <INPUT_FILE> [--outfile FILE] "
                  << "[--length SEC] [--round-interval SEC] [--fps N] [--maxrecords N]\n";
        exit(1);
    }

    return cfg;
}

std::vector<std::string> collect_files(const std::string &path) {
    std::vector<std::string> files;

    if (std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.path().extension() == ".h5") {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(path);
    }
    return files;
}

int run_streaming_mode(const Config &cfg, const std::vector<std::string> &files,
                       const std::string &first_file, int number_of_frames, int round_freq_s) {
    std::cout << "\n=== STREAMING MODE ===\n";

    std::chrono::high_resolution_clock::time_point t0, t1;

    // Take round_interval from config or compute it
    int64_t round_interval = cfg.round_interval_s;
    if (round_interval <= 0) {
        if (number_of_frames <= 0) {
            std::cerr << "Error: cannot compute round interval without --length or --round-interval\n";
            return 1;
        }
        // Get global timestamp range
        t0 = std::chrono::high_resolution_clock::now();
        auto [global_min, global_max] = get_global_timestamp_range(files, cfg.dataset_name);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Get min/max timestamp " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

        // Compute round interval
        auto duration = global_max - global_min;
        round_interval = duration / number_of_frames;
        if (round_interval < round_freq_s) round_interval = round_freq_s;
        std::cout << "Computed round interval: " << round_interval << "s\n\n";
    }


    // Stream and aggregate files
    t0 = std::chrono::high_resolution_clock::now();
    auto result = stream_aggregate_files(files, cfg.dataset_name, round_interval, cfg.chunk_size);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Stream aggregation: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    // Write results
    t0 = std::chrono::high_resolution_clock::now();
    write_results(cfg.outfile, result.grouped, result.total_seen, result.added_stats);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Write results: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    // Copy metadata
    copy_metadata(first_file, cfg.outfile, static_cast<int>(round_interval),
                  static_cast<int>(result.max_vehicle_id),
                  result.max_unique_vehicle_count, result.max_unique_on_segment_count);

    return 0;
}

int run_legacy_mode(const Config &cfg, const std::vector<std::string> &files,
                    const std::string &first_file, int number_of_frames, int round_freq_s) {

    // ------------- Load all data into memory -------------
    std::vector<VehicleData> data;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto& f : files) {
        std::cout << "Loading: " << f << "\n";
        read_into_memory(f, cfg.dataset_name, data, cfg.max_records, round_freq_s);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Read all HDF5 files: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    // ------------- Process data -------------
    std::vector<std::pair<size_t, size_t>> ranges;
    t0 = std::chrono::high_resolution_clock::now();
    auto round_interval = round_and_chunk(data, number_of_frames, round_freq_s, ranges, cfg.round_interval_s);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Round+Chunk: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    // ------------- Count unique vehicles -------------
    t0 = std::chrono::high_resolution_clock::now();
    auto vehicles_count = static_cast<int>(get_max_vehicle_id(data));
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Count unique vehicles: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    // ------------- Parallel aggregation -------------
    int nthreads = omp_get_max_threads();
    std::vector<GroupedMap> grouped_local(nthreads);
    std::vector<SeenMap> seen_local(nthreads);

    t0 = std::chrono::high_resolution_clock::now();
    aggregate_parallel(data, ranges, grouped_local, seen_local);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel aggregation: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    t0 = std::chrono::high_resolution_clock::now();
    auto added_stats = compute_added_stats(data, vehicles_count);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Compute added stats: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    data.clear();

    GroupedMap grouped;
    std::unordered_map<FrameID, int64_t> total_seen;
    int max_unique_vehicle_count = 0, max_unique_on_segment_count = 0;

    t0 = std::chrono::high_resolution_clock::now();
    merge_thread_results(grouped_local, seen_local, grouped,
                         max_unique_vehicle_count, max_unique_on_segment_count,
                         total_seen);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Merge thread-local maps: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    grouped_local.clear();
    seen_local.clear();


    // ------------- Write results -------------
    t0 = std::chrono::high_resolution_clock::now();
    write_results(cfg.outfile, grouped, total_seen, added_stats);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Build and write results: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    grouped.clear();
    total_seen.clear();
    added_stats.clear();

    t0 = std::chrono::high_resolution_clock::now();
    copy_metadata(first_file, cfg.outfile, static_cast<int>(round_interval), vehicles_count,
                  max_unique_vehicle_count, max_unique_on_segment_count);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Copy file-level attributes: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    return 0;
}

int main(int argc, char* argv[]) {
    auto t_start = std::chrono::high_resolution_clock::now();

    Config cfg = parseArgs(argc, argv);

    const int number_of_frames = (cfg.length_s > 0) ? (cfg.length_s * cfg.fps) : 0;
    int round_freq_s = 5; // will be read from file

    std::cout << "Input file: " << cfg.filename << "\n";
    std::cout << "Dataset name: " << cfg.dataset_name << "\n";
    std::cout << "Output file: " << cfg.outfile << "\n";
    if (cfg.length_s > 0) {
        std::cout << "Length (s): " << cfg.length_s << "\n";
        std::cout << "FPS: " << cfg.fps << "\n";
        std::cout << "Number of frames: " << number_of_frames << "\n";
    } else {
        std::cout << "Using explicit round interval (s): " << cfg.round_interval_s << "\n";
    }
    std::cout << "Streaming mode: " << (cfg.streaming_mode ? "enabled" : "disabled") << "\n\n";

    auto files = collect_files(cfg.filename);
    if (files.empty()) {
        std::cerr << "No .h5 files found to process.\n";
        return 1;
    }

    std::string first_file = files[0];
    std::cout << "Found " << files.size() << " file(s) to process\n";

    int result;
    if (cfg.streaming_mode) {
        result = run_streaming_mode(cfg, files, first_file, number_of_frames, round_freq_s);
    } else {
        result = run_legacy_mode(cfg, files, first_file, number_of_frames, round_freq_s);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "\nTotal time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms\n";
    std::cout << "Output written to: " << cfg.outfile << "\n";

    return result;
}

