#pragma once

#include <vector>
#include <string>
#include <tuple>

// Log function declaration
void log(const std::string& message);

namespace ruthlib {
    void setup_ace(int n_threads);
    void setup_map(std::string path);
    void init_routes();
    bool is_master();
    void do_alternatives(std::vector<std::pair<int, int>> OD_matrix, int max_routes, bool use_origin_speeds = false);
    std::tuple<std::vector<int>, std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<float>>> get_routes();
    void do_travel_times(std::vector<std::vector<int>> routes);
    std::vector<float> get_travel_times();
    void update_speeds(std::vector<std::pair<int, float>> edge_speeds);
    void barrier();
    std::vector<std::pair<int, int>> load_od_matrix(const std::string& filepath, int line_limit = 3);
    void finalize();
}
