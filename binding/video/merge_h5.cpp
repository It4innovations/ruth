#include "H5Cpp.h"
#include <vector>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <execution>
#include <fstream>
#include <sstream>
using namespace H5;
namespace fs = std::filesystem;

struct VehicleData {
    int64_t timestamp;
    int64_t node_from;
    int64_t node_to;
    int32_t segment_length;
    int64_t vehicle_id;
    float start_offset_m;
    float speed_mps;
    bool active;
};

CompType get_vehicle_dtype() {
    CompType mtype(sizeof(VehicleData));
    mtype.insertMember("timestamp", HOFFSET(VehicleData, timestamp), PredType::NATIVE_INT64);
    mtype.insertMember("node_from", HOFFSET(VehicleData, node_from), PredType::NATIVE_INT64);
    mtype.insertMember("node_to", HOFFSET(VehicleData, node_to), PredType::NATIVE_INT64);
    mtype.insertMember("segment_length", HOFFSET(VehicleData, segment_length), PredType::NATIVE_INT32);
    mtype.insertMember("vehicle_id", HOFFSET(VehicleData, vehicle_id), PredType::NATIVE_INT64);
    mtype.insertMember("start_offset_m", HOFFSET(VehicleData, start_offset_m), PredType::NATIVE_FLOAT);
    mtype.insertMember("speed_mps", HOFFSET(VehicleData, speed_mps), PredType::NATIVE_FLOAT);
    mtype.insertMember("active", HOFFSET(VehicleData, active), PredType::NATIVE_HBOOL);
    return mtype;
}

void copy_attributes(const DataSet &src, DataSet &dst) {
    for (int i = 0; i < src.getNumAttrs(); ++i) {
        Attribute attr = src.openAttribute(i);
        std::string name = attr.getName();
        std::cout << "Copying attribute: " << name << "\n";
        DataType dtype = attr.getDataType();
        DataSpace dspace = attr.getSpace();
        Attribute dst_attr = dst.createAttribute(name, dtype, dspace);
        std::vector<char> buffer(attr.getInMemDataSize());
        attr.read(dtype, buffer.data());
        dst_attr.write(dtype, buffer.data());
    }
}

void copy_file_attributes(const H5File &src, const H5File &dst) {
    for (int i = 0; i < src.getNumAttrs(); ++i) {
        Attribute attr = src.openAttribute(i);
        std::string name = attr.getName();
        std::cout << "Copying file attribute: " << name << "\n";
        DataType dtype = attr.getDataType();
        DataSpace dspace = attr.getSpace();
        Attribute dst_attr = dst.createAttribute(name, dtype, dspace);
        std::vector<char> buffer(attr.getInMemDataSize());
        attr.read(dtype, buffer.data());
        dst_attr.write(dtype, buffer.data());
    }
}

void read_h5(const std::string &filename, const std::string &dataset_name,
             std::vector<VehicleData> &out) {
    try {
        H5File file(filename, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(dataset_name);
        DataSpace dataspace = dataset.getSpace();
        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims);
        std::vector<VehicleData> tmp(dims[0]);
        dataset.read(tmp.data(), get_vehicle_dtype());
        out.insert(out.end(), tmp.begin(), tmp.end());
        std::cout << "Read " << dims[0] << " records from " << filename << "\n";
    } catch (const FileIException &e) {
        std::cerr << "Error reading " << filename << ": " << e.getCDetailMsg() << "\n";
    }
}

auto to_datetime(int64_t timestamp) {
    std::time_t t = static_cast<time_t>(timestamp);
    std::tm tm = *std::gmtime(&t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buf);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <folder>\n";
        return 1;
    }

    const std::string folder = argv[1];
    const std::string dataset_name = "fcd";
    std::vector<VehicleData> all_data;
    std::vector<std::string> files;

    for (const auto &entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".h5") files.push_back(entry.path().string());
    }

    if (files.empty()) {
        std::cerr << "No .h5 files found in folder: " << folder << "\n";
        return 1;
    }

    std::sort(files.begin(), files.end());
    std::cout << "Merging " << files.size() << " files...\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto &f : files) read_h5(f, dataset_name, all_data);
    std::cout << "Total records loaded: " << all_data.size() << "\n";

    std::sort(std::execution::par, all_data.begin(), all_data.end(),
              [](const VehicleData &a, const VehicleData &b) {
                  return a.timestamp < b.timestamp;
              });

    if (!all_data.empty()) {
        std::cout << "First timestamp: " << all_data.front().timestamp
                  << " (" << to_datetime(all_data.front().timestamp) << ")\n";
        std::cout << "Last timestamp: " << all_data.back().timestamp
                  << " (" << to_datetime(all_data.back().timestamp) << ")\n";
    }

    H5File outfile("merged.h5", H5F_ACC_TRUNC);
    hsize_t dims[1] = {static_cast<hsize_t>(all_data.size())};
    DataSpace dataspace(1, dims);
    DataSet dataset = outfile.createDataSet(dataset_name, get_vehicle_dtype(), dataspace);
    dataset.write(all_data.data(), get_vehicle_dtype());

    // Copy attributes from first file
    try {
        H5File src(files.front(), H5F_ACC_RDONLY);
        copy_file_attributes(src, outfile);
        copy_attributes(src.openDataSet(dataset_name), dataset);
        std::cout << "Copied attributes from " << files.front() << "\n";
    } catch (const Exception &e) {
        std::cerr << "Warning: failed to copy attributes — " << e.getDetailMsg() << "\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Merged dataset written in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms.\n";
}
