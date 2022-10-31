#pragma once

#include <vector>
#include <cstdint>

using Edge = uint32_t;
using Node = uint32_t;

struct Graph {
    std::vector<std::vector<Edge>> nodes;
};

struct RoutingResult {
    std::vector<std::vector<uint32_t>> routes;
};

extern "C" {
    Graph* init_graph();
    Edge add_node(Graph* graph);
    void add_edge(Graph* graph, Edge from, Edge to);

    RoutingResult* perform_routing(Graph* graph, Node from, Node to, int alternatives);
    uint32_t get_route_length(RoutingResult* result, int route_index);
    uint32_t get_route_node(RoutingResult* result, int route_index, int node_index);
    void free_routing(RoutingResult* result);
}
