
// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "lib.cpp"  // Include the implementation
#include "lib.hpp"

namespace py = pybind11;

template<typename T>
py::array_t<T> vec_to_numpy(std::vector<T> vec) {
    auto* ptr = new std::vector<T>(std::move(vec));
    auto capsule = py::capsule(ptr, [](void* p) {
        delete reinterpret_cast<std::vector<T>*>(p);
    });
    return py::array_t<T>(
        {static_cast<py::ssize_t>(ptr->size())},
        {sizeof(T)},
        ptr->data(),
        capsule
    );
}

PYBIND11_MODULE(ruthlib, m) {
    m.doc() = "pybind11 bindings for the ruth library";


    m.def("setup_map", &ruthlib::setup_map, "setup the map of the simulation");
    m.def("setup_ace", &ruthlib::setup_ace, "setup the ace library (MPI is included in ace)");
    m.def("do_alternatives",
        [](py::array_t<int, py::array::c_style | py::array::forcecast> od, int max_routes,
           bool use_origin_speeds) {
            auto buf = od.request();
            if (buf.ndim != 2 || buf.shape[1] != 2)
                throw std::runtime_error("OD matrix must be shape (N, 2)");
            ruthlib::do_alternatives(static_cast<const int*>(buf.ptr),
                                     static_cast<size_t>(buf.shape[0]),
                                     max_routes,
                                     use_origin_speeds);
        },
        py::arg("od"),
        py::arg("max_routes"),
        py::arg("use_origin_speeds") = false,
        "perform alternative routes calculation");

    m.def("is_master", &ruthlib::is_master, "checks if the rank is the master");
    m.def("finalize", &ruthlib::finalize, "performs ace teardown (include MPI finalize");
    m.def("barrier", &ruthlib::barrier, "insert a strong synch between ranks. MUST be called by all ranks");
    m.def("load_od_matrix", &ruthlib::load_od_matrix, "load od matrix from file");
    m.def("init_routes", &ruthlib::init_routes, "initialize state for routes. to be called only once");
    m.def("get_routes", []() {
        auto* flat = new ruthlib::RoutesFlat(ruthlib::get_routes_flat());
        auto capsule = py::capsule(flat, [](void* p) {
            delete reinterpret_cast<ruthlib::RoutesFlat*>(p);
        });
        auto make_array = [&](auto* data, size_t n) {
            using T = std::remove_pointer_t<decltype(data)>;
            return py::array_t<T>({static_cast<py::ssize_t>(n)}, {sizeof(T)}, data, capsule);
        };
        py::dict result;
        result["vehicle_ids"]   = make_array(flat->vehicle_ids.data(),   flat->vehicle_ids.size());
        result["travel_times"]  = make_array(flat->travel_times.data(),  flat->travel_times.size());
        result["route_offsets"] = make_array(flat->route_offsets.data(), flat->route_offsets.size());
        result["node_offsets"]  = make_array(flat->node_offsets.data(),  flat->node_offsets.size());
        result["nodes"]         = make_array(flat->nodes.data(),         flat->nodes.size());
        return result;
    }, "returns alternative routes as a dict of zero-copy numpy arrays (CSR format)");

    m.def("get_travel_times", []() {
        return vec_to_numpy(ruthlib::get_travel_times());
    }, "returns travel times as a zero-copy numpy float32 array");

    m.def("do_travel_times", &ruthlib::do_travel_times, "calculate travel times for the routes");
    m.def("update_speeds",
        [](py::array_t<int,   py::array::c_style | py::array::forcecast> ids,
           py::array_t<float, py::array::c_style | py::array::forcecast> spds) {
            auto bid = ids.request(), bsp = spds.request();
            if (bid.ndim != 1 || bsp.ndim != 1 || bid.shape[0] != bsp.shape[0])
                throw std::runtime_error("edge_ids and speeds must be 1-D arrays of equal length");
            ruthlib::update_speeds(static_cast<const int*>(bid.ptr),
                                   static_cast<const float*>(bsp.ptr),
                                   static_cast<size_t>(bid.shape[0]));
        }, "update speeds of edges in the graph");
    m.def("is_simulation_running", &ruthlib::is_simulation_running, "check if the simulation is running");
    m.def("finish_simulation", &ruthlib::finish_simulation, "finish the simulation and clean up resources");
}
