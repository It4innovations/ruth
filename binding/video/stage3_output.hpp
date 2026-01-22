#pragma once

#include "common_types.hpp"
#include "stage1_loading.hpp"
#include "stage2_aggregation.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>

// ==================== STAGE 3: OUTPUT GENERATION ====================
// This stage converts aggregated data to HDF5 format and writes results

// Flattened row for HDF5 output
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

// Index row for fast frame lookup
struct IndexRow {
    FrameID frame_id;
    int64_t start_index;
    int64_t end_index;
    int64_t active_vehicles;
    int64_t vehicles_finished;
    int64_t total_distance;
    float total_time;
};

namespace hdf5_output {
inline H5::CompType create_grouped_row_type() {
    H5::CompType mtype(sizeof(GroupedRow));
    mtype.insertMember("rounded_ts", HOFFSET(GroupedRow, frame_id), H5::PredType::NATIVE_INT64);
    mtype.insertMember("node_from", HOFFSET(GroupedRow, node_from), H5::PredType::NATIVE_INT64);
    mtype.insertMember("node_to", HOFFSET(GroupedRow, node_to), H5::PredType::NATIVE_INT64);
    mtype.insertMember("speed_sum_first", HOFFSET(GroupedRow, speed_sum_first), H5::PredType::NATIVE_DOUBLE);
    mtype.insertMember("speed_count_first", HOFFSET(GroupedRow, speed_count_first), H5::PredType::NATIVE_INT32);
    mtype.insertMember("count_unique_first", HOFFSET(GroupedRow, count_unique_first), H5::PredType::NATIVE_INT32);
    mtype.insertMember("speed_sum_second", HOFFSET(GroupedRow, speed_sum_second), H5::PredType::NATIVE_DOUBLE);
    mtype.insertMember("speed_count_second", HOFFSET(GroupedRow, speed_count_second), H5::PredType::NATIVE_INT32);
    mtype.insertMember("count_unique_second", HOFFSET(GroupedRow, count_unique_second), H5::PredType::NATIVE_INT32);
    return mtype;
}

inline H5::CompType create_index_row_type() {
    H5::CompType itype(sizeof(IndexRow));
    itype.insertMember("rounded_ts", HOFFSET(IndexRow, frame_id), H5::PredType::NATIVE_INT64);
    itype.insertMember("start_index", HOFFSET(IndexRow, start_index), H5::PredType::NATIVE_INT64);
    itype.insertMember("end_index", HOFFSET(IndexRow, end_index), H5::PredType::NATIVE_INT64);
    itype.insertMember("active_vehicles", HOFFSET(IndexRow, active_vehicles), H5::PredType::NATIVE_INT64);
    itype.insertMember("vehicles_finished", HOFFSET(IndexRow, vehicles_finished), H5::PredType::NATIVE_INT64);
    itype.insertMember("total_distance", HOFFSET(IndexRow, total_distance), H5::PredType::NATIVE_INT64);
    itype.insertMember("total_time", HOFFSET(IndexRow, total_time), H5::PredType::NATIVE_FLOAT);
    return itype;
}
}

// Flatten grouped map to vector of rows
inline std::vector<GroupedRow> flatten_grouped(const GroupedMap& grouped) {
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

// Build index from grouped rows
inline std::vector<IndexRow> build_index(const std::vector<GroupedRow> &rows,
    const std::unordered_map<FrameID, int64_t> &total_seen_vehicles,
    const std::unordered_map<FrameID, AddedStats> &added_stats) {

    std::vector<IndexRow> index;
    if (rows.empty()) return index;
    index.reserve(total_seen_vehicles.size());

    auto current_frame_id = rows[0].frame_id;
    int64_t start = 0;
    int64_t vehicles_finished = 0;
    int64_t total_distance = 0;
    float total_time = 0.0f;

    for (size_t i = 0; i < rows.size(); i++) {
        if (rows[i].frame_id != current_frame_id) {
            auto end_index = i;
            auto seen_it = total_seen_vehicles.find(current_frame_id);
            auto active_vehicles = (seen_it != total_seen_vehicles.end()) ? seen_it->second : 0;

            auto stats_it = added_stats.find(current_frame_id);
            const auto delta_distance = (stats_it != added_stats.end()) ? stats_it->second.distance : 0.0f;
            const auto delta_time = (stats_it != added_stats.end()) ? stats_it->second.time : 0;

            total_distance += static_cast<int64_t>(delta_distance);
            total_time += static_cast<float>(delta_time);

            index.emplace_back(IndexRow{current_frame_id,
                start, static_cast<int64_t>(end_index),
                active_vehicles, vehicles_finished,
                total_distance, total_time});
            current_frame_id = rows[i].frame_id;
            start = static_cast<int64_t>(i);
        }
        vehicles_finished += rows[i].vehicles_finished;
    }

    // last segment
    size_t end_index = rows.size();
    auto seen_it = total_seen_vehicles.find(current_frame_id);
    auto stats_it = added_stats.find(current_frame_id);

    const auto active_vehicles = (seen_it != total_seen_vehicles.end()) ? seen_it->second : 0;
    const auto delta_distance = (stats_it != added_stats.end()) ? stats_it->second.distance : 0.0f;
    const auto delta_time = (stats_it != added_stats.end()) ? stats_it->second.time : 0;

    total_distance += static_cast<int64_t>(delta_distance);
    total_time += static_cast<float>(delta_time);

    index.emplace_back(IndexRow{current_frame_id, start, static_cast<int64_t>(end_index),
        active_vehicles, vehicles_finished, total_distance, total_time});

    return index;
}

// Write grouped data to HDF5
inline void write_grouped_to_hdf5(const std::string &filename,
                                  const std::string &dataset_name,
                                  const std::vector<GroupedRow> &rows) {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::CompType mtype = hdf5_output::create_grouped_row_type();
    hsize_t dims[1] = { rows.size() };
    H5::DataSpace dataspace(1, dims);
    H5::DSetCreatPropList plist;
    hsize_t chunk_dims[1] = { std::min<hsize_t>(100000, rows.size()) };
    if (chunk_dims[0] > 0) plist.setChunk(1, chunk_dims);
    H5::DataSet dataset = file.createDataSet(dataset_name, mtype, dataspace, plist);
    dataset.write(rows.data(), mtype);
}

// Write index to HDF5
inline void write_index_to_hdf5(const std::string &filename,
                                const std::string &dataset_name,
                                const std::vector<IndexRow> &index) {
    H5::H5File file(filename, H5F_ACC_RDWR);
    H5::CompType itype = hdf5_output::create_index_row_type();
    hsize_t dims[1] = { index.size() };
    H5::DataSpace space(1, dims);
    H5::DataSet dset = file.createDataSet(dataset_name, itype, space);
    dset.write(index.data(), itype);
}

// Copy file attributes with additional metadata
inline void copy_file_attributes(const H5::H5File& src, const H5::H5File& dst,
                                 const std::unordered_map<std::string, std::string> &additional_info) {
    int n_attrs = src.getNumAttrs();
    for (int i = 0; i < n_attrs; ++i) {
        H5::Attribute attr = src.openAttribute(i);
        std::string name = attr.getName();
        H5::DataType dtype = attr.getDataType();
        H5::DataSpace dspace = attr.getSpace();
        H5::Attribute dst_attr = dst.createAttribute(name, dtype, dspace);
        std::vector<char> buffer(attr.getInMemDataSize());
        attr.read(dtype, buffer.data());
        dst_attr.write(dtype, buffer.data());
    }
    for (const auto& [name, value] : additional_info) {
        H5::StrType strdatatype(H5::PredType::C_S1, H5T_VARIABLE);
        hsize_t dim[] = {1};
        H5::DataSpace attr_dataspace(1, dim);
        H5::Attribute new_attr = dst.createAttribute(name, strdatatype, attr_dataspace);
        new_attr.write(strdatatype, value);
    }
}

// Copy metadata from source to destination file
inline void copy_metadata(const std::string &src, const std::string &dst,
                          int round_interval, int vehicle_count,
                          int max_unique_vehicles_in_time_segment,
                          int max_unique_vehicles_on_segment) {
    std::unordered_map<std::string, std::string> info{
        {"interval_s", std::to_string(round_interval)},
        {"number_of_vehicles", std::to_string(vehicle_count)},
        {"max_unique_vehicles_in_time_segment", std::to_string(max_unique_vehicles_in_time_segment)},
        {"max_unique_vehicles_on_segment", std::to_string(max_unique_vehicles_on_segment)}
    };
    H5::H5File src_file(src, H5F_ACC_RDONLY);
    H5::H5File dst_file(dst, H5F_ACC_RDWR);
    copy_file_attributes(src_file, dst_file, info);
}

// Write complete aggregated results to HDF5
inline void write_results(const std::string &outfile,
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

// Read functions for querying output
inline std::vector<IndexRow> read_index(const std::string &filename, const std::string &dataset_name) {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::CompType itype(sizeof(IndexRow));
    itype.insertMember("rounded_ts", HOFFSET(IndexRow, frame_id), H5::PredType::NATIVE_INT64);
    itype.insertMember("start_index", HOFFSET(IndexRow, start_index), H5::PredType::NATIVE_INT64);
    itype.insertMember("end_index", HOFFSET(IndexRow, end_index), H5::PredType::NATIVE_INT64);
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1]; dataspace.getSimpleExtentDims(dims);
    std::vector<IndexRow> out(dims[0]);
    dataset.read(out.data(), itype);
    return out;
}

inline void read_rows_for_range(const std::string &filename,
                                const std::string &dataset_name,
                                int64_t start_index, int64_t count,
                                std::vector<GroupedRow> &out_rows) {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::CompType mtype = hdf5_output::create_grouped_row_type();
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t count_arr[1] = { static_cast<hsize_t>(count) };
    hsize_t offset[1] = { static_cast<hsize_t>(start_index) };
    dataspace.selectHyperslab(H5S_SELECT_SET, count_arr, offset);
    H5::DataSpace memspace(1, count_arr);
    out_rows.resize(count);
    dataset.read(out_rows.data(), mtype, memspace, dataspace);
}

inline bool lookup_index(const std::vector<IndexRow> &index, int64_t rounded_ts,
                         int64_t &start_out, int64_t &count_out) {
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
