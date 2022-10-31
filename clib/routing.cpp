#include "routing.h"

#include <stdexcept>

extern "C" {
    Graph* init_graph() {
        return new Graph{};
    }
    Edge add_node(Graph* graph) {
        auto id = graph->nodes.size();
        graph->nodes.emplace_back();
        return static_cast<Edge>(id);
    }
    void add_edge(Graph* graph, Edge from, Edge to) {
        if (from >= graph->nodes.size() || to >= graph->nodes.size()) {
            throw std::runtime_error{"Wrong edge"};
        }
        graph->nodes[from].push_back(to);
    }

    RoutingResult* perform_routing(Graph* graph, Node from, Node to, int alternatives) {
        auto* result = new RoutingResult{};
        for (int i = 0; i < alternatives; i++) {
            result->routes.push_back({from, static_cast<Node>(i + 1), to});
        }
        return result;
    }
    uint32_t get_route_length(RoutingResult* result, int route_index) {
        return result->routes[route_index].size();
    }
    uint32_t get_route_node(RoutingResult* result, int route_index, int node_index) {
        return result->routes[route_index][node_index];
    }
    void free_routing(RoutingResult* result) {
        delete result;
    }
}
