/*
    A C++ program that reads simulation output fcd_history.h5,
    processes it to aggregate statistics over time segments and road segments,
    and writes the aggregated results back to a new HDF5 file which is used for video generation.

    The program uses parallel processing with OpenMP to speed up the aggregation step.
*/

#include "H5Cpp.h"
#include <vector>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <execution>
#include <numeric>
#include <getopt.h>
#include <filesystem>

using Timestamp = int64_t;
using FrameID = int64_t;
using VehicleID = int64_t;

struct TimeSegmentKey{
    FrameID frame_id;
    int64_t node_from;
    int64_t node_to;

    bool operator==(const TimeSegmentKey &other) const{
        return frame_id == other.frame_id &&
               node_from == other.node_from &&
               node_to == other.node_to;
    }
};

inline void hash_combine(std::size_t &seed, std::size_t value){
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <>
struct std::hash<TimeSegmentKey>{
    size_t operator()(const TimeSegmentKey &k) const noexcept {
        size_t seed = 54;
        hash_combine(seed, std::hash<int64_t>()(k.frame_id));
        hash_combine(seed, std::hash<int64_t>()(k.node_from));
        hash_combine(seed, std::hash<int64_t>()(k.node_to));
        return seed;
    }
};


// --------------------------------------------------

struct AggregatedTimeSegmentData{
    int vehicles_finished = 0;

    double speed_sum_first = 0.0;
    int speed_count_first = 0;
    int count_unique_first = 0;

    double speed_sum_second = 0.0;
    int speed_count_second = 0;
    int count_unique_second = 0;

    void add_speed(double speed, bool pos) {
        if (pos) {
            speed_sum_first += speed; speed_count_first++;
        } else {
            speed_sum_second += speed; speed_count_second++;
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

struct VehicleData{
    FrameID frame_id = 0;
    Timestamp timestamp;
    int64_t node_from;
    int64_t node_to;
    int32_t segment_length;
    VehicleID vehicle_id;
    float start_offset_m;
    float speed_mps;
    bool active;
};

struct GroupedRow {
    FrameID frame_id;
    int64_t node_from;
    int64_t node_to;

    double speed_sum_first;
    int speed_count_first;
    int count_unique_first;

    double speed_sum_second;
    int speed_count_second;
    int count_unique_second;

    int vehicles_finished;
};

struct IndexRow {
    FrameID frame_id;
    int64_t start_index;
    int64_t end_index;
    int64_t active_vehicles;
    int64_t vehicles_finished;
    int64_t total_distance;
    float total_time;
};

struct AddedStats {
    int64_t time = 0;
    float distance = 0.0f;
};

using GroupedMap = std::unordered_map<TimeSegmentKey, AggregatedTimeSegmentData>;
using SeenMap    = std::unordered_map<FrameID, std::unordered_set<VehicleID>>;
using namespace H5;

void read_into_memory(const std::string &filename, const std::string &dataset_name,
                      std::vector<VehicleData> &data, hsize_t max_records, int &round_freq_s) {

    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: file '" << filename << "' does not exist\n";
        return;
    }

    H5File file(filename, H5F_ACC_RDONLY);
    if (H5Lexists(file.getId(), dataset_name.c_str(), H5P_DEFAULT) <= 0) {
        std::cerr << "Error: dataset '" << dataset_name << "' not found in " << filename << "\n";
        return;
    }

    DataSet dataset = file.openDataSet(dataset_name);
    CompType mtype(sizeof(VehicleData));
    mtype.insertMember("timestamp", HOFFSET(VehicleData, timestamp), PredType::NATIVE_INT64);
    mtype.insertMember("node_from", HOFFSET(VehicleData, node_from), PredType::NATIVE_INT64);
    mtype.insertMember("node_to", HOFFSET(VehicleData, node_to), PredType::NATIVE_INT64);
    mtype.insertMember("segment_length", HOFFSET(VehicleData, segment_length), PredType::NATIVE_INT32);
    mtype.insertMember("vehicle_id", HOFFSET(VehicleData, vehicle_id), PredType::NATIVE_INT64);
    mtype.insertMember("start_offset_m", HOFFSET(VehicleData, start_offset_m), PredType::NATIVE_FLOAT);
    mtype.insertMember("speed_mps", HOFFSET(VehicleData, speed_mps), PredType::NATIVE_FLOAT);
    mtype.insertMember("active", HOFFSET(VehicleData, active), PredType::NATIVE_HBOOL);

    DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1]; dataspace.getSimpleExtentDims(dims);
    hsize_t count[1] = { std::min(dims[0], max_records) };

    size_t current_size = data.size();
    data.resize(current_size + count[0]);

    hsize_t offset[1] = {0};
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
    DataSpace memspace(1, count);
    dataset.read(data.data() + current_size, mtype, memspace, dataspace);

    if (H5Aexists(file.getId(), "round_freq_s") > 0) {
        Attribute attr = file.openAttribute("round_freq_s");
        attr.read(PredType::NATIVE_INT, &round_freq_s);
    }
}

int64_t round_and_chunk(std::vector<VehicleData> &data, const int number_of_frames, const int round_freq_s,
                     std::vector<std::pair<size_t,size_t>> &ranges, const int64_t explicit_round_interval = -1){

    const Timestamp first_ts = data.front().timestamp;
    const Timestamp last_ts = data.back().timestamp;
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

    std::ranges::sort(data, [](const VehicleData &a, const VehicleData &b) {
        if (a.timestamp != b.timestamp) return a.timestamp > b.timestamp;
        if (a.node_from != b.node_from) return a.node_from < b.node_from;
        if (a.node_to != b.node_to) return a.node_to < b.node_to;
        return a.start_offset_m > b.start_offset_m;
    });

    size_t begin = 0;
    size_t chunk_size = 1'000'000;
    for (size_t i = 1; i < data.size(); i++){
        if (data[i].frame_id != data[i-1].frame_id && (i - begin) > chunk_size){
            ranges.emplace_back(begin, i);
            begin = i;
        }
    }
    ranges.emplace_back(begin, data.size());

    return round_interval;
}

std::vector<GroupedRow> flatten_grouped(const GroupedMap& grouped) {
    std::vector<GroupedRow> rows;
    rows.reserve(grouped.size());
    for (const auto& [key, agg] : grouped) {
        rows.emplace_back(GroupedRow{
            key.frame_id,
            key.node_from,
            key.node_to,
            agg.speed_sum_first,
            agg.speed_count_first,
            agg.count_unique_first,
            agg.speed_sum_second,
            agg.speed_count_second,
            agg.count_unique_second,
            agg.vehicles_finished
        });
    }
    return rows;
}

// Copy file-level attributes
void copy_file_attributes(const H5File& src, const H5File& dst, const std::unordered_map<std::string, std::string> &additional_info = {}) {
    int n_attrs = src.getNumAttrs();
    for (int i = 0; i < n_attrs; ++i) {
        Attribute attr = src.openAttribute(i);
        std::string name = attr.getName();
        DataType dtype = attr.getDataType();
        DataSpace dspace = attr.getSpace();
        Attribute dst_attr = dst.createAttribute(name, dtype, dspace);

        std::vector<char> buffer(attr.getInMemDataSize());
        attr.read(dtype, buffer.data());
        dst_attr.write(dtype, buffer.data());
    }

    for (const auto& [name, value] : additional_info) {
        // Create a string attribute
        StrType strdatatype(PredType::C_S1, H5T_VARIABLE);
        hsize_t dim[] = {1};
        DataSpace attr_dataspace(1, dim);
        Attribute new_attr = dst.createAttribute(name, strdatatype, attr_dataspace);
        new_attr.write(strdatatype, value);
    }
}

void write_grouped_to_hdf5(const std::string &filename,
                           const std::string &dataset_name,
                           const std::vector<GroupedRow> &rows) {
    using namespace H5;
    H5File file(filename, H5F_ACC_TRUNC);

    // Describe the compound type
    CompType mtype(sizeof(GroupedRow));
    mtype.insertMember("rounded_ts", HOFFSET(GroupedRow, frame_id), PredType::NATIVE_INT64);
    mtype.insertMember("node_from", HOFFSET(GroupedRow, node_from), PredType::NATIVE_INT64);
    mtype.insertMember("node_to", HOFFSET(GroupedRow, node_to), PredType::NATIVE_INT64);
    mtype.insertMember("speed_sum_first", HOFFSET(GroupedRow, speed_sum_first), PredType::NATIVE_DOUBLE);
    mtype.insertMember("speed_count_first", HOFFSET(GroupedRow, speed_count_first), PredType::NATIVE_INT32);
    mtype.insertMember("count_unique_first", HOFFSET(GroupedRow, count_unique_first), PredType::NATIVE_INT32);
    mtype.insertMember("speed_sum_second", HOFFSET(GroupedRow, speed_sum_second), PredType::NATIVE_DOUBLE);
    mtype.insertMember("speed_count_second", HOFFSET(GroupedRow, speed_count_second), PredType::NATIVE_INT32);
    mtype.insertMember("count_unique_second", HOFFSET(GroupedRow, count_unique_second), PredType::NATIVE_INT32);
    // note: vehicles_finished not included in dataset, only accumulated in index


    // Create dataspace
    hsize_t dims[1] = { rows.size() };
    DataSpace dataspace(1, dims);

    // Optional: set chunking by timestamp (e.g., 100k rows per chunk)
    DSetCreatPropList plist;
    hsize_t chunk_dims[1] = { std::min<hsize_t>(100000, rows.size()) };
    plist.setChunk(1, chunk_dims);

    // Create dataset and write
    DataSet dataset = file.createDataSet(dataset_name, mtype, dataspace, plist);
    dataset.write(rows.data(), mtype);
}

std::vector<IndexRow> build_index(const std::vector<GroupedRow> &rows,
    const std::unordered_map<FrameID, int64_t> &total_seen_vehicles,
    const std::unordered_map<FrameID, AddedStats> &added_stats) {
    // total_seen and added_stats are separate, because added_stats cannot be computed in parallel

    std::vector<IndexRow> index;
    if (rows.empty()) return index;

    auto current_frame_id = rows[0].frame_id;
    int64_t start = 0;
    int64_t vehicles_finished = 0;
    int64_t total_distance = 0;
    float total_time = 0.0f;

    for (int i = 0; i < rows.size(); i++) {
        if (rows[i].frame_id != current_frame_id) {
            auto end_index = i;
            auto active_vehicles = total_seen_vehicles.at(current_frame_id);

            const auto delta_distance = added_stats.at(current_frame_id).distance;
            const auto delta_time = added_stats.at(current_frame_id).time;
            total_distance += delta_distance;
            total_time += delta_time;

            index.emplace_back(IndexRow{current_frame_id,
                start, end_index,
                active_vehicles, vehicles_finished,
                total_distance, total_time});
            current_frame_id = rows[i].frame_id;
            start = i;
        }
        vehicles_finished += rows[i].vehicles_finished;
    }

    // last segment
    int end_index = rows.size();
    const auto active_vehicles = total_seen_vehicles.at(current_frame_id);

    const auto delta_distance = added_stats.at(current_frame_id).distance;
    total_distance += delta_distance;

    const auto delta_time = added_stats.at(current_frame_id).time;
    total_time += delta_time;

    index.emplace_back(IndexRow{current_frame_id, start, end_index,
        active_vehicles, vehicles_finished, total_distance, total_time});

    return index;
}

VehicleID get_unique_vehicle_count(const std::vector<VehicleData> &data)
{
    VehicleID vehicle_id_max = 0;
    std::unordered_set<int64_t> unique_vehicles;
    for (const auto &d : data) {
        unique_vehicles.insert(d.vehicle_id);
        if (d.vehicle_id > vehicle_id_max) vehicle_id_max = d.vehicle_id;
    }
    std::cout << "\t Unique vehicles in data: " << unique_vehicles.size() << "\n";
    std::cout << "\t Max vehicle ID: " << vehicle_id_max << "\n";

    return vehicle_id_max;
}

void write_index_to_hdf5(const std::string &filename,
                         const std::string &dataset_name,
                         const std::vector<IndexRow> &index) {
    H5File file(filename, H5F_ACC_RDWR);
    CompType itype(sizeof(IndexRow));
    itype.insertMember("rounded_ts", HOFFSET(IndexRow, frame_id), PredType::NATIVE_INT64);
    itype.insertMember("start_index", HOFFSET(IndexRow, start_index), PredType::NATIVE_INT64);
    itype.insertMember("end_index", HOFFSET(IndexRow, end_index), PredType::NATIVE_INT64);
    itype.insertMember("active_vehicles", HOFFSET(IndexRow, active_vehicles), PredType::NATIVE_INT64);
    itype.insertMember("vehicles_finished", HOFFSET(IndexRow, vehicles_finished), PredType::NATIVE_INT64);
    itype.insertMember("total_distance", HOFFSET(IndexRow, total_distance), PredType::NATIVE_INT64);
    itype.insertMember("total_time", HOFFSET(IndexRow, total_time), PredType::NATIVE_FLOAT);

    hsize_t dims[1] = { index.size() };
    DataSpace space(1, dims);
    DataSet dset = file.createDataSet(dataset_name, itype, space);
    dset.write(index.data(), itype);
}

// --------------------------------------------------
std::vector<IndexRow> read_index(const std::string &filename, const std::string &dataset_name) {
    H5File file(filename, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet(dataset_name);
    CompType itype(sizeof(IndexRow));
    itype.insertMember("rounded_ts", HOFFSET(IndexRow, frame_id), PredType::NATIVE_INT64);
    itype.insertMember("start_index", HOFFSET(IndexRow, start_index), PredType::NATIVE_INT64);
    itype.insertMember("end_index", HOFFSET(IndexRow, end_index), PredType::NATIVE_INT64);

    DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1]; dataspace.getSimpleExtentDims(dims);
    std::vector<IndexRow> out(dims[0]);
    dataset.read(out.data(), itype);
    return out;
}

bool lookup_index(const std::vector<IndexRow> &index, int64_t rounded_ts, int64_t &start_out, int64_t &count_out) {
    size_t lo = 0, hi = index.size();
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (index[mid].frame_id == rounded_ts) {
            start_out = index[mid].start_index;
            count_out = index[mid].end_index;
            return true;
        } else if (index[mid].frame_id < rounded_ts) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return false;
}

// ----------------------- read rows for a known offset/count -----------------------
void read_rows_for_range(const std::string &filename,
                         const std::string &dataset_name,
                         int64_t start_index, int64_t count,
                         std::vector<GroupedRow> &out_rows) {
    H5File file(filename, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet(dataset_name);

    CompType mtype(sizeof(GroupedRow));
    mtype.insertMember("rounded_ts", HOFFSET(GroupedRow, frame_id), PredType::NATIVE_INT64);
    mtype.insertMember("node_from", HOFFSET(GroupedRow, node_from), PredType::NATIVE_INT64);
    mtype.insertMember("node_to", HOFFSET(GroupedRow, node_to), PredType::NATIVE_INT64);
    mtype.insertMember("speed_sum_first", HOFFSET(GroupedRow, speed_sum_first), PredType::NATIVE_DOUBLE);
    mtype.insertMember("speed_count_first", HOFFSET(GroupedRow, speed_count_first), PredType::NATIVE_INT32);
    mtype.insertMember("count_unique_first", HOFFSET(GroupedRow, count_unique_first), PredType::NATIVE_INT32);
    mtype.insertMember("speed_sum_second", HOFFSET(GroupedRow, speed_sum_second), PredType::NATIVE_DOUBLE);
    mtype.insertMember("speed_count_second", HOFFSET(GroupedRow, speed_count_second), PredType::NATIVE_INT32);
    mtype.insertMember("count_unique_second", HOFFSET(GroupedRow, count_unique_second), PredType::NATIVE_INT32);

    DataSpace dataspace = dataset.getSpace();
    hsize_t count_arr[1] = { static_cast<hsize_t>(count) };
    hsize_t offset[1] = { static_cast<hsize_t>(start_index) };

    dataspace.selectHyperslab(H5S_SELECT_SET, count_arr, offset);
    DataSpace memspace(1, count_arr);

    out_rows.resize(count);
    dataset.read(out_rows.data(), mtype, memspace, dataspace);
}
// --------------------------------------------------

void aggregate_parallel(const std::vector<VehicleData> &data,
                        const std::vector<std::pair<size_t, size_t>> &ranges,
                        std::vector<GroupedMap> &grouped_local,
                        std::vector<SeenMap> &seen_local) {
#pragma omp parallel for schedule(dynamic) default(none) shared(data, ranges, grouped_local, seen_local)
    for (auto [begin, end] : ranges) {
        int thread_id = omp_get_thread_num();
        for (size_t i = begin; i < end; ++i) {
            const auto &d = data[i];
            int64_t frame_id = d.frame_id;
            TimeSegmentKey key{frame_id, d.node_from, d.node_to};

            auto &agg = grouped_local[thread_id][key];
            bool pos = (d.start_offset_m / d.segment_length) < 0.5f;
            agg.add_speed(d.speed_mps, pos);

            if (seen_local[thread_id][frame_id].insert(d.vehicle_id).second)
                agg.increment_count_unique(pos);

            if (!d.active)
                agg.increment_vehicles_finished();
        }
    }
}

std::unordered_map<FrameID, AddedStats> compute_added_stats(const std::vector<VehicleData> &data, int vehicles_count) {
    std::vector<int> future_vehicle_info(vehicles_count + 1, -1);
    std::unordered_map<FrameID, AddedStats> added_stats{};

    const auto earliest_timestamp = data.back().timestamp;
    for (size_t i = 0; i < data.size(); ++i) {
        const auto &fcd = data[i];
        auto vehicle_idx = fcd.vehicle_id;

        if (future_vehicle_info[vehicle_idx] != -1) {
            int future_idx = future_vehicle_info[vehicle_idx];
            const auto &fut = data[future_idx];
            auto delta_time = fut.timestamp - fcd.timestamp;
            if (delta_time < 0)
                throw std::runtime_error("Timestamps not sorted correctly!");
            added_stats[fut.frame_id].time += delta_time;
            if (fut.node_from == fcd.node_from && fut.node_to == fcd.node_to)
                added_stats[fut.frame_id].distance += fut.start_offset_m - fcd.start_offset_m;
            else
                added_stats[fut.frame_id].distance += fut.start_offset_m;
        }
        future_vehicle_info[vehicle_idx] = i;
    }

    // Handle first timestamp vehicles
    for (int info_index : future_vehicle_info) {
        if (info_index != -1) {
            const auto &fut = data[info_index];
            auto delta_time = fut.timestamp - earliest_timestamp;
            if (delta_time < 0)
                throw std::runtime_error("Timestamps not sorted correctly!");
            added_stats[fut.frame_id].time += delta_time;
            added_stats[fut.frame_id].distance += fut.start_offset_m;
        }
    }
    return added_stats;
}

void merge_thread_results(std::vector<GroupedMap> &grouped_local,
                          std::vector<SeenMap> &seen_local,
                          GroupedMap &grouped,
                          int &max_unique_vehicle_count,
                          int &max_unique_on_segment_count,
                          std::unordered_map<FrameID, int64_t> &total_seen) {
    int nthreads = grouped_local.size();
    grouped.reserve(std::accumulate(grouped_local.begin(), grouped_local.end(), 0ul,
                                    [](size_t s, const auto &m) { return s + m.size(); }));

    for (int t = 0; t < nthreads; ++t) {
        for (auto &[key, agg] : grouped_local[t]) {
            if (auto it = grouped.find(key); it != grouped.end())
                throw std::runtime_error("Key collision during merge!");
            grouped.emplace(key, agg);

            int unique_count = seen_local[t][key.frame_id].size();
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
    for (int t = 0; t < nthreads; ++t)
        for (const auto &[frame_id, ids] : seen_local[t])
            total_seen[frame_id] += ids.size();
}

void write_results(const std::string &outfile,
                   const GroupedMap &grouped,
                   const std::unordered_map<FrameID, int64_t> &total_seen,
                   const std::unordered_map<FrameID, AddedStats> &added_stats) {
    auto rows = flatten_grouped(grouped);
    std::ranges::sort(rows, [](const GroupedRow &a, const GroupedRow &b) {
        if (a.frame_id != b.frame_id) return a.frame_id < b.frame_id;
        if (a.node_from != b.node_from) return a.node_from < b.node_from;
        if (a.node_to != b.node_to) return a.node_to < b.node_to;
        return a.count_unique_first < b.count_unique_first;
    });

    write_grouped_to_hdf5(outfile, "grouped_data", rows);
    auto idx = build_index(rows, total_seen, added_stats);
    write_index_to_hdf5(outfile, "grouped_index", idx);
}

void copy_metadata(const std::string &src, const std::string &dst,
                   int round_interval, int vehicle_count,
                   int max_unique_vehicles_in_time_segment,
                   int max_unique_vehicles_on_segment) {
    std::unordered_map<std::string, std::string> info{
            {"interval_s", std::to_string(round_interval)},
            {"number_of_vehicles", std::to_string(vehicle_count)},
            {"max_unique_vehicles_in_time_segment", std::to_string(max_unique_vehicles_in_time_segment)},
            {"max_unique_vehicles_on_segment", std::to_string(max_unique_vehicles_on_segment)}
    };
    H5File src_file(src, H5F_ACC_RDONLY);
    H5File dst_file(dst, H5F_ACC_RDWR);
    copy_file_attributes(src_file, dst_file, info);
}

struct Config {
    std::string filename;
    std::string dataset_name = "fcd";
    std::string outfile = "fcd_aggregated.h5";
    int length_s = -1;
    int fps = 25;
    size_t max_records = std::numeric_limits<size_t>::max();
    int round_interval_s = -1;
};

Config parseArgs(int argc, char* argv[]) {
    Config cfg;

    static struct option long_options[] = {
        {"outfile", required_argument, nullptr, 'o'},
        {"length", required_argument, nullptr, 'l'},
        {"round-interval", required_argument, nullptr, 'r'},
        {"fps", required_argument, nullptr, 'p'},
        {"maxrecords", required_argument, nullptr, 'm'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "o:l:p:m:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 'o': cfg.outfile = optarg; break;
            case 'l': cfg.length_s = std::stoi(optarg); break;
            case 'r': cfg.round_interval_s = std::stoi(optarg); break;
            case 'p': cfg.fps = std::stoi(optarg); break;
            case 'm': cfg.max_records = std::stoull(optarg); break;
            case 'h':
                std::cout << "Usage: ./video_preprocess <INPUT_FILE> [--outfile FILE] "
                             "[--length SEC] [--round-interval SEC] [--fps N] [--maxrecords N]\n";
                exit(0);
        }
    }

    if (optind < argc) {
        cfg.filename = argv[optind];
    } else {
        std::cerr << "Error: filename is required\n";
        exit(1);
    }

    // Require that user provides either duration (--length) or explicit round interval (--round-interval)
    if (cfg.length_s <= 0 && cfg.round_interval_s <= 0) {
        std::cerr << "Error: you must specify either --length <seconds> or --round-interval <seconds>\n";
        std::cerr << "Usage: ./video_preprocess <INPUT_FILE> [--outfile FILE] "
                  << "[--length SEC] [--round-interval SEC] [--fps N] [--maxrecords N]\n";
        exit(1);
    }

    return cfg;
}

int main(int argc, char* argv[]) {
    auto t_start = std::chrono::high_resolution_clock::now();

    Config cfg = parseArgs(argc, argv);

    const std::string &filename = cfg.filename;
    const std::string &dataset_name = cfg.dataset_name;
    const std::string &outfile = cfg.outfile;
    const int length_s = cfg.length_s;
    const int fps = cfg.fps;
    const size_t max_records = cfg.max_records;
    const int number_of_frames = (length_s > 0) ? (length_s * fps) : 0;
    int round_freq_s = 5; // will be read from file

    std::cout << "Input file: " << filename << "\n";
    std::cout << "Dataset name: " << dataset_name << "\n";
    std::cout << "Output file: " << outfile << "\n";
    if (length_s > 0) {
        std::cout << "Length (s): " << length_s << "\n";
        std::cout << "FPS: " << fps << "\n\n";
        std::cout << "Number of frames: " << number_of_frames << "\n";
    } else {
        std::cout << "Using explicit round interval (s): " << cfg.round_interval_s << "\n\n";
    }

// --------------------- Read input data --------------------
    std::vector<VehicleData> data;
    std::vector<std::string> files_to_process;
    std::string first_file = "";

    if (std::filesystem::is_directory(filename)) {
        for (const auto& entry : std::filesystem::directory_iterator(filename)) {
            if (entry.path().extension() == ".h5") {
                files_to_process.push_back(entry.path().string());
            }
        }
    } else {
        files_to_process.push_back(filename);
    }

    if (files_to_process.empty()) {
        std::cerr << "No .h5 files found to process.\n";
        return 1;
    }
    first_file = files_to_process[0];

    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto& f : files_to_process) {
        std::cout << "Loading: " << f << "\n";
        read_into_memory(f, dataset_name, data, max_records, round_freq_s);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Read all HDF5 files: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
    // ----------------------------------------------------------

    // ----------------- Round timestamps and chunk data --------
    std::vector<std::pair<size_t,size_t>> ranges;
    t0 = std::chrono::high_resolution_clock::now();
    auto round_interval = round_and_chunk(data, number_of_frames, round_freq_s, ranges, cfg.round_interval_s);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Round+Chunk: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
    // ----------------------------------------------------------


    // ----------------- Count unique vehicles ------------------
    t0 = std::chrono::high_resolution_clock::now();
    int vehicles_count = get_unique_vehicle_count(data);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Count unique vehicles: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";


    // ------------------ Parallel aggregation ----------------
    int nthreads = omp_get_max_threads();
    std::vector<GroupedMap> grouped_local(nthreads);
    std::vector<SeenMap> seen_local(nthreads);

    t0 = std::chrono::high_resolution_clock::now();
    aggregate_parallel(data, ranges, grouped_local, seen_local);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel aggregation: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
    // ----------------------------------------------------------

    // ------------------ Compute added stats ----------------
    t0 = std::chrono::high_resolution_clock::now();
    auto added_stats = compute_added_stats(data, vehicles_count);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Compute added stats: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
    // ----------------------------------------------------------
    data.clear();


    // ----------------- Merge thread-local maps ----------------

    GroupedMap grouped;
    std::unordered_map<FrameID, int64_t> total_seen;
    int max_unique_vehicle_count = 0, max_unique_on_segment_count = 0;

    t0 = std::chrono::high_resolution_clock::now();
    merge_thread_results(grouped_local, seen_local, grouped,
                         max_unique_vehicle_count, max_unique_on_segment_count,
                         total_seen);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Merge thread-local maps: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
    // ----------------------------------------------------------
    grouped_local.clear();
    seen_local.clear();

    // ----------------- Build and write index ----------------
    t0 = std::chrono::high_resolution_clock::now();
    write_results(outfile, grouped, total_seen, added_stats);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Build and write results: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
    grouped.clear();
    total_seen.clear();
    added_stats.clear();

    // ----------------- Copy file-level attributes ----------------
    t0 = std::chrono::high_resolution_clock::now();
    copy_metadata(first_file, outfile, round_interval, vehicles_count,
                  max_unique_vehicle_count, max_unique_on_segment_count);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Copy file-level attributes: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
    // ----------------------------------------------------------

    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t_start).count() << " ms\n";
    std::cout << "Output written to: " << outfile << "\n";

    return 0;
}