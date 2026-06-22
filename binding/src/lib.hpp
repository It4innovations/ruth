#pragma once

#include <string>
#include <utility>
#include <vector>

namespace ruthlib {

void setup_map(std::string path);
void setup_ace(int n_threads);

void do_alternatives(const int* od_data, size_t n_pairs, int max_routes, bool use_origin_speeds = false);
void do_alternatives(std::vector<std::pair<int, int>> OD_matrix, int max_routes, bool use_origin_speeds = false);
void do_travel_times(std::vector<std::vector<int>> routes);

bool is_master();
void barrier();
void finalize();

std::vector<std::pair<int, int>> load_od_matrix(const std::string &filepath,
                                                 int line_limit);
void init_routes();

struct RoutesFlat {
    std::vector<int>   vehicle_ids;
    std::vector<float> travel_times;
    std::vector<int>   route_offsets;  // length V+1
    std::vector<int>   node_offsets;   // length total_routes+1
    std::vector<int>   nodes;          // flat node ids
};
RoutesFlat get_routes_flat();

std::vector<float> get_travel_times();
void update_speeds(const int* edge_ids, const float* speeds, size_t n);
void update_speeds(std::vector<std::pair<int, float>> edge_speeds);
bool is_simulation_running();
void finish_simulation();

} // namespace ruthlib
