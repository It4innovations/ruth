import cffi

if __name__ == "__main__":
    libname = "librouting.so"

    ffi = cffi.FFI()

    # Needs `pip install cffi`
    # Needs to be kept in sync with routing.h
    # Cannot use preprocessor or C++ things
    ffi.cdef("""
        typedef void* Graph;

        // Graph
        Graph init_graph();
        uint32_t add_node(Graph);
        void add_edge(Graph, uint32_t, uint32_t);

        // Routing
        typedef void* RoutingResult;

        RoutingResult perform_routing(Graph graph, uint32_t from, uint32_t to, int alternatives);
        uint32_t get_route_length(RoutingResult result, int route_index);
        uint32_t get_route_node(RoutingResult result, int route_index, int node_index);
        void free_routing(RoutingResult result);
    """)
    lib = ffi.dlopen(libname)
    graph = lib.init_graph()
    a = lib.add_node(graph)
    b = lib.add_node(graph)
    lib.add_edge(graph, a, b)

    alternatives = 5
    routes = lib.perform_routing(graph, a, b, alternatives)

    for route_index in range(alternatives):
        node_count = lib.get_route_length(routes, route_index)
        route = [lib.get_route_node(routes, route_index, node_index) for node_index in range(node_count)]
        print(f"Route #{route_index}: {route}")

    lib.free_routing(routes)
