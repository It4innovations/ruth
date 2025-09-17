
// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lib.cpp"  // Include the implementation

namespace py = pybind11;

PYBIND11_MODULE(ruthlib, m) {
    m.doc() = "pybind11 bindings for the ruth library";
    
    m.def("setup_map", &ruthlib::setup_map, "setup the map of the simulation");
    m.def("setup_ace", &ruthlib::setup_ace, "setup the ace library (MPI is included in ace)");
    m.def("do_alternatives", &ruthlib::do_alternatives, "perform alternative routes calculation");
    m.def("is_master", &ruthlib::is_master, "checks if the rank is the master");
    m.def("finalize", &ruthlib::finalize, "performs ace teardown (include MPI finalize");
    m.def("barrier", &ruthlib::barrier, "insert a strong synch between ranks. MUST be called by all ranks");
    m.def("load_od_matrix", &ruthlib::load_od_matrix, "load od matrix from file");
    m.def("init_routes", &ruthlib::init_routes, "initialize state for routes. to be called only once");
    m.def("get_routes", &ruthlib::get_routes, "returns alternative routes to python as tuple");
    m.def("get_travel_times", &ruthlib::get_travel_times, "returns travel times to python as vector");
    m.def("do_travel_times", &ruthlib::do_travel_times, "calculate travel times for the routes");
    m.def("update_speeds", &ruthlib::update_speeds, "update speeds of edges in the graph");
    m.def("is_simulation_running", &ruthlib::is_simulation_running, "check if the simulation is running");
    m.def("finish_simulation", &ruthlib::finish_simulation, "finish the simulation and clean up resources");
}

