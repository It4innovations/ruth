#include <iostream>
#include <string>
#include <thread>
#include "ruthlib.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --od-matrix PATH     Path to OD matrix CSV file (default: dataset/OD_matrix.csv)\n"
              << "  --map PATH           Path to map HDF5 file (default: dataset/map_prague.hdf5)\n"
              << "  --num-vehicles N     Number of vehicles to process (default: 10000)\n"
              << "  --threads N          Number of threads to use (default: 128)\n"
              << "  -h, --help          Show this help message\n";
}

int main(int argc, char *argv[]) {
    // Default values
    int num_vehicles = 10000;
    int num_threads = std::thread::hardware_concurrency();
    std::string od_matrix_path = "dataset/OD_matrix.csv";
    std::string map_path = "dataset/map_prague.hdf5";
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--od-matrix" && i + 1 < argc) {
            od_matrix_path = argv[++i];
        } else if (arg == "--map" && i + 1 < argc) {
            map_path = argv[++i];
        } else if (arg == "--num-vehicles" && i + 1 < argc) {
            num_vehicles = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    


    ruthlib::setup_ace(num_threads);

    // std::cout << "Configuration:" << std::endl;
    // std::cout << "  OD Matrix: " << od_matrix_path << std::endl;
    // std::cout << "  Map: " << map_path << std::endl;
    // std::cout << "  Vehicles: " << num_vehicles << std::endl;
    // std::cout << "  Threads: " << num_threads << std::endl;

    ruthlib::init_routes();

    auto start_time = std::chrono::high_resolution_clock::now();
    if (ruthlib::is_master()) {
        
        std::vector<std::pair<int, int>> OD_matrix = ruthlib::load_od_matrix(
            od_matrix_path, num_vehicles);

        start_time = std::chrono::high_resolution_clock::now();

        ruthlib::setup_map(map_path);

        ruthlib::do_alternatives(OD_matrix, 1);

        auto end_time_alt = std::chrono::high_resolution_clock::now();
        auto elapsed_alt = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_alt - start_time).count();
        std::cout << "Alternatives Elapsed time: " << elapsed_alt << " ms" << std::endl;

        auto [vehicle_ids, routes_per_vehicle, travel_times_per_vehicle] = ruthlib::get_routes();

        // Print all the nodes in routes_per_vehicle[0]
        std::cout << "Routes for vehicles:" << routes_per_vehicle.size() << " routes" << std::endl;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_routes = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - end_time_alt).count();
        std::cout << "Elapsed route time: " << elapsed_routes << " ms" << std::endl;

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;

        log("All workloads completed, exiting...");
    }

    ruthlib::finalize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;
} // main
