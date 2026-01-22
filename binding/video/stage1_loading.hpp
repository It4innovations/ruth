#pragma once

#include "common_types.hpp"
#include <H5Cpp.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

// ==================== STAGE 1: DATA LOADING ====================
// This stage handles reading raw vehicle data from HDF5 files

struct FileTimestampMeta {
    std::string filepath;
    Timestamp min_ts = std::numeric_limits<Timestamp>::max();
    Timestamp max_ts = std::numeric_limits<Timestamp>::min();
    uint64_t record_count = 0;
};

namespace hdf5_helpers {
inline void validate_dataset(const H5::DataSet &dataset, const std::string &dataset_name) {
    if (dataset.getTypeClass() != H5T_COMPOUND)
        throw std::runtime_error("Dataset '" + dataset_name + "' is not a compound type");
    H5::DataSpace dataspace = dataset.getSpace();
    if (dataspace.getSimpleExtentNdims() != 1)
        throw std::runtime_error("Dataset '" + dataset_name + "' must be 1D");
    H5::CompType ctype = dataset.getCompType();
    const std::vector<std::string> required = {
        "timestamp", "node_from", "node_to", "segment_length",
        "vehicle_id", "start_offset_m", "speed_mps", "active"
    };
    for (const auto &name : required) {
        if (ctype.getMemberIndex(name.c_str()) < 0)
            throw std::runtime_error("Dataset '" + dataset_name + "' missing field '" + name + "'");
    }
}

inline H5::CompType create_vehicle_data_type() {
    H5::CompType mtype(sizeof(VehicleData));
    mtype.insertMember("timestamp", HOFFSET(VehicleData, timestamp), H5::PredType::NATIVE_INT64);
    mtype.insertMember("node_from", HOFFSET(VehicleData, node_from), H5::PredType::NATIVE_INT64);
    mtype.insertMember("node_to", HOFFSET(VehicleData, node_to), H5::PredType::NATIVE_INT64);
    mtype.insertMember("segment_length", HOFFSET(VehicleData, segment_length), H5::PredType::NATIVE_INT32);
    mtype.insertMember("vehicle_id", HOFFSET(VehicleData, vehicle_id), H5::PredType::NATIVE_INT64);
    mtype.insertMember("start_offset_m", HOFFSET(VehicleData, start_offset_m), H5::PredType::NATIVE_FLOAT);
    mtype.insertMember("speed_mps", HOFFSET(VehicleData, speed_mps), H5::PredType::NATIVE_FLOAT);
    mtype.insertMember("active", HOFFSET(VehicleData, active), H5::PredType::NATIVE_INT8);
    return mtype;
}
}

// Chunked reader for streaming mode
class ChunkedHDF5Reader {
public:
    ChunkedHDF5Reader(const std::string &filename, const std::string &dataset_name, uint64_t chunk_size)
        : filename_(filename), dataset_name_(dataset_name), chunk_size_(chunk_size),
          current_offset_(0), total_records_(0), exhausted_(false) {
        file_ = std::make_unique<H5::H5File>(filename, H5F_ACC_RDONLY);
        dataset_ = std::make_unique<H5::DataSet>(file_->openDataSet(dataset_name));
        hdf5_helpers::validate_dataset(*dataset_, dataset_name_);
        H5::DataSpace dataspace = dataset_->getSpace();
        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims);
        total_records_ = dims[0];
    }

    std::vector<VehicleData> read_next_chunk() {
        if (exhausted_ || current_offset_ >= total_records_) {
            exhausted_ = true;
            return {};
        }
        uint64_t remaining = total_records_ - current_offset_;
        uint64_t count = std::min(chunk_size_, remaining);
        std::vector<VehicleData> chunk(count);
        H5::CompType mtype = hdf5_helpers::create_vehicle_data_type();
        H5::DataSpace dataspace = dataset_->getSpace();
        hsize_t offset_arr[1] = {current_offset_};
        hsize_t count_arr[1] = {count};
        dataspace.selectHyperslab(H5S_SELECT_SET, count_arr, offset_arr);
        H5::DataSpace memspace(1, count_arr);
        try {
            dataset_->read(chunk.data(), mtype, memspace, dataspace);
        } catch (const H5::Exception &ex) {
            throw std::runtime_error("Failed to read chunk at offset " + std::to_string(current_offset_) +
                                     " from '" + filename_ + "': " + ex.getDetailMsg());
        }
        current_offset_ += count;
        if (current_offset_ >= total_records_) exhausted_ = true;
        return chunk;
    }

    bool is_exhausted() const { return exhausted_; }
    uint64_t total_records() const { return total_records_; }
    const std::string& filename() const { return filename_; }

private:
    std::string filename_;
    std::string dataset_name_;
    uint64_t chunk_size_;
    uint64_t current_offset_;
    uint64_t total_records_;
    bool exhausted_;
    std::unique_ptr<H5::H5File> file_;
    std::unique_ptr<H5::DataSet> dataset_;
};

// Scan file metadata without loading all data
inline FileTimestampMeta scan_file_metadata(const std::string &filename, const std::string &dataset_name) {
    FileTimestampMeta meta; meta.filepath = filename;
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("Error: file '" + filename + "' does not exist");
    }
    H5::H5File file(filename, H5F_ACC_RDONLY);
    if (H5Lexists(file.getId(), dataset_name.c_str(), H5P_DEFAULT) <= 0) {
        throw std::runtime_error("Error: dataset '" + dataset_name + "' not found in " + filename);
    }
    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1]; dataspace.getSimpleExtentDims(dims);
    meta.record_count = dims[0];
    if (dims[0] == 0) return meta;
    H5::CompType ts_type(sizeof(Timestamp));
    ts_type.insertMember("timestamp", 0, H5::PredType::NATIVE_INT64);
    const size_t sample_count = std::min<size_t>(100, dims[0]);
    for (size_t i = 0; i < sample_count; ++i) {
        hsize_t idx = (i * (dims[0] - 1)) / (sample_count > 1 ? sample_count - 1 : 1);
        hsize_t one = 1; hsize_t offset[1] = {idx};
        dataspace.selectHyperslab(H5S_SELECT_SET, &one, offset);
        H5::DataSpace memspace(1, &one);
        Timestamp ts; dataset.read(&ts, ts_type, memspace, dataspace);
        meta.min_ts = std::min(meta.min_ts, ts);
        meta.max_ts = std::max(meta.max_ts, ts);
    }
    return meta;
}

// Get global timestamp range across all files
inline std::pair<Timestamp, Timestamp> get_global_timestamp_range(
    const std::vector<std::string> &files, const std::string &dataset_name) {
    Timestamp global_min = std::numeric_limits<Timestamp>::max();
    Timestamp global_max = std::numeric_limits<Timestamp>::min();
    for (const auto &file : files) {
        auto meta = scan_file_metadata(file, dataset_name);
        global_min = std::min(global_min, meta.min_ts);
        global_max = std::max(global_max, meta.max_ts);
        std::cout << "  File: " << file << "  ts range: [" << meta.min_ts << ", " << meta.max_ts
                  << "]  records: " << meta.record_count << "\n";
    }
    return {global_min, global_max};
}

// Load entire file into memory (legacy mode)
inline void read_into_memory(const std::string &filename, const std::string &dataset_name,
                             std::vector<VehicleData> &data, uint64_t max_records, int &round_freq_s) {
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("Error: file '" + filename + "' does not exist");
    }
    H5::H5File file(filename, H5F_ACC_RDONLY);
    if (H5Lexists(file.getId(), dataset_name.c_str(), H5P_DEFAULT) <= 0) {
        throw std::runtime_error("Error: dataset '" + dataset_name + "' not found in " + filename);
    }
    H5::DataSet dataset = file.openDataSet(dataset_name);
    hdf5_helpers::validate_dataset(dataset, dataset_name);
    H5::CompType mtype = hdf5_helpers::create_vehicle_data_type();
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1]; dataspace.getSimpleExtentDims(dims);
    hsize_t count[1] = { std::min(dims[0], static_cast<hsize_t>(max_records)) };
    size_t current_size = data.size();
    data.resize(current_size + count[0]);
    hsize_t offset[1] = {0};
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
    H5::DataSpace memspace(1, count);
    try {
        dataset.read(data.data() + current_size, mtype, memspace, dataspace);
    } catch (const H5::Exception &ex) {
        throw std::runtime_error("Failed to read dataset '" + dataset_name + "' from '" + filename +
                                 "': " + ex.getDetailMsg());
    }
    if (H5Aexists(file.getId(), "round_freq_s") > 0) {
        H5::Attribute attr = file.openAttribute("round_freq_s");
        attr.read(H5::PredType::NATIVE_INT, &round_freq_s);
    }
}
