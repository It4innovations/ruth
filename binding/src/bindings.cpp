
// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include "lib.cpp"  // Include the implementation
#include "lib.hpp"
#include "advance_types.hpp"
#include "advance_orchestration.hpp"
#include "advance_store.hpp"

namespace py = pybind11;

// Reinterpret a 1-D structured-dtype numpy array as a flat AdvanceTask buffer.
// The Python dtype must match the C++ struct layout exactly; the itemsize check
// (with advance_task_itemsize() asserted at import) guards against drift.
static std::vector<ruthlib::AdvanceTask> buffer_to_advance_tasks(const py::array& arr) {
  py::buffer_info b = arr.request();
  if (static_cast<size_t>(b.itemsize) != sizeof(ruthlib::AdvanceTask)) {
    throw std::runtime_error("AdvanceTask dtype itemsize does not match the C++ struct");
  }
  const size_t n = static_cast<size_t>(b.size);
  std::vector<ruthlib::AdvanceTask> tasks(n);
  if (n) std::memcpy(tasks.data(), b.ptr, n * sizeof(ruthlib::AdvanceTask));
  return tasks;
}

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
    m.def("setup_ace", &ruthlib::setup_ace,
        py::arg("n_threads"), py::arg("queue") = "prioritized",
        "setup the ace library (MPI is included in ace). queue selects the scheduler "
        "workload queue: \"prioritized\" (default) or \"fifo\".");
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

    m.def("do_travel_times",
        [](std::vector<std::vector<int>> routes) {
            py::gil_scoped_release release;  // blocks on the dispatch barrier
            ruthlib::do_travel_times(std::move(routes));
        },
        "calculate travel times for the routes");
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
    m.def("run_worker_loop",
        []() {
            py::gil_scoped_release release;  // blocks for the whole run in the barrier loop
            ruthlib::run_worker_loop();
        },
        "worker rank driver: pair every master barrier until the simulation ends "
        "(barrier-based BSP; enables ACE work-stealing)");

    // ----------------------------------------------------------------------
    // Distributed vehicle-advance
    // ----------------------------------------------------------------------

    // Size of one AdvanceTask, so Python can assert its structured dtype's
    // itemsize matches the C++ struct layout at import time.
    m.def("advance_task_itemsize", []() {
        return static_cast<size_t>(sizeof(ruthlib::AdvanceTask));
    }, "byte size of one AdvanceTask (for dtype validation)");

    // Phase 1: scatter free-vehicle chunks + whole queues and wait. Tasks are
    // passed as 1-D structured-dtype arrays (see ADVANCE_TASK_DTYPE in Python);
    // queue_offsets is a CSR delimiter of length num_queues+1 into queue_tasks.
    m.def("do_advance",
        [](const py::array& free_tasks, const py::array& queue_tasks,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> queue_offsets,
           int64_t departure_epoch_us, double los_tolerance_s) {
            std::vector<ruthlib::AdvanceTask> free_vec = buffer_to_advance_tasks(free_tasks);
            std::vector<ruthlib::AdvanceTask> queue_vec = buffer_to_advance_tasks(queue_tasks);

            auto qo = queue_offsets.request();
            if (qo.ndim != 1)
                throw std::runtime_error("queue_offsets must be a 1-D array");
            const int32_t* off = static_cast<const int32_t*>(qo.ptr);
            const size_t n_queues = qo.shape[0] > 0 ? static_cast<size_t>(qo.shape[0]) - 1 : 0;

            std::vector<std::vector<ruthlib::AdvanceTask>> queues;
            queues.reserve(n_queues);
            for (size_t q = 0; q < n_queues; ++q) {
                queues.emplace_back(queue_vec.begin() + off[q], queue_vec.begin() + off[q + 1]);
            }

            py::gil_scoped_release release;  // blocks on MPI sends + waits
            ruthlib::do_advance(free_vec, queues, departure_epoch_us, los_tolerance_s);
        },
        py::arg("free_tasks"), py::arg("queue_tasks"), py::arg("queue_offsets"),
        py::arg("departure_epoch_us"), py::arg("los_tolerance_s"),
        "phase 1: evaluate free vehicles and queues across ranks");

    // Prune every rank's GV replica to datetime >= threshold_epoch_s. Call at
    // the start of each step (before do_advance) to match serial's read-time GV.
    m.def("advance_drop_old",
        [](double threshold_epoch_s) {
            py::gil_scoped_release release;  // blocks on the broadcast
            ruthlib::advance_drop_old(threshold_epoch_s);
        },
        py::arg("threshold_epoch_s"),
        "prune every GV replica to datetime >= threshold (before do_advance)");

    // Phase 2: broadcast the sync workload; every rank enters sync_allgatherv
    // and merges the step's FCDs into its GV replica. (Pruning is separate,
    // see advance_drop_old.)
    m.def("sync_advance",
        []() {
            py::gil_scoped_release release;  // blocks on the collective
            ruthlib::sync_advance();
        },
        "phase 2: all-gather the step's FCDs into every GV replica");

    m.def("advance_add_local_fcds",
        [](py::array_t<double,  py::array::c_style | py::array::forcecast> dt,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> vid,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> sid,
           py::array_t<double,  py::array::c_style | py::array::forcecast> off,
           py::array_t<double,  py::array::c_style | py::array::forcecast> spd) {
            auto bdt = dt.request(), bvid = vid.request(), bsid = sid.request(),
                 boff = off.request(), bspd = spd.request();
            const size_t n = static_cast<size_t>(bdt.shape[0]);
            if (bvid.shape[0] != (py::ssize_t)n || bsid.shape[0] != (py::ssize_t)n ||
                boff.shape[0] != (py::ssize_t)n || bspd.shape[0] != (py::ssize_t)n)
                throw std::runtime_error("advance_add_local_fcds: array lengths differ");
            ruthlib::advance_add_local_fcds(
                static_cast<const double*>(bdt.ptr), static_cast<const int32_t*>(bvid.ptr),
                static_cast<const int64_t*>(bsid.ptr), static_cast<const double*>(boff.ptr),
                static_cast<const double*>(bspd.ptr), n);
        },
        py::arg("datetime_s"), py::arg("vehicle_id"), py::arg("segment_id"),
        py::arg("offset_m"), py::arg("speed_mps"),
        "inject stationary FCDs into this rank's accumulator before sync_advance");

    m.def("get_advance_deltas", []() {
        auto* flat = new ruthlib::AdvanceDeltasFlat(ruthlib::get_advance_deltas());
        auto capsule = py::capsule(flat, [](void* p) {
            delete reinterpret_cast<ruthlib::AdvanceDeltasFlat*>(p);
        });
        auto make_array = [&](auto* data, size_t n) {
            using T = std::remove_pointer_t<decltype(data)>;
            return py::array_t<T>({static_cast<py::ssize_t>(n)}, {sizeof(T)}, data, capsule);
        };
        py::dict r;
        r["vehicle_id"]      = make_array(flat->vehicle_id.data(),        flat->vehicle_id.size());
        r["new_start_index"] = make_array(flat->new_start_index.data(),   flat->new_start_index.size());
        r["new_offset_m"]    = make_array(flat->new_offset_m.data(),      flat->new_offset_m.size());
        r["new_time_offset_us"] = make_array(flat->new_time_offset_us.data(), flat->new_time_offset_us.size());
        r["moved"]           = make_array(flat->moved.data(),             flat->moved.size());
        r["changed_segment"] = make_array(flat->changed_segment.data(),   flat->changed_segment.size());
        r["reached_end"]     = make_array(flat->reached_end.data(),       flat->reached_end.size());
        return r;
    }, "per-vehicle movement deltas from the last sync (dict of zero-copy arrays)");

    m.def("get_advance_fcds", []() {
        auto* flat = new ruthlib::AdvanceFCDsFlat(ruthlib::get_advance_fcds());
        auto capsule = py::capsule(flat, [](void* p) {
            delete reinterpret_cast<ruthlib::AdvanceFCDsFlat*>(p);
        });
        auto make_array = [&](auto* data, size_t n) {
            using T = std::remove_pointer_t<decltype(data)>;
            return py::array_t<T>({static_cast<py::ssize_t>(n)}, {sizeof(T)}, data, capsule);
        };
        py::dict r;
        r["datetime_seconds"]   = make_array(flat->datetime_seconds.data(),   flat->datetime_seconds.size());
        r["vehicle_id"]         = make_array(flat->vehicle_id.data(),         flat->vehicle_id.size());
        r["segment_id"]         = make_array(flat->segment_id.data(),         flat->segment_id.size());
        r["offset_from_start"]  = make_array(flat->offset_from_start.data(),  flat->offset_from_start.size());
        r["vehicle_speed_mps"]  = make_array(flat->vehicle_speed_mps.data(),  flat->vehicle_speed_mps.size());
        return r;
    }, "FCDs from the last sync for the history (dict of zero-copy arrays)");

    // --- VehicleStore (shadow mode) -----------------------------------------
    //
    // These exist so the store can be populated alongside the Python Vehicle
    // objects and its task construction compared against theirs, every step,
    // before anything depends on it. Nothing in the simulation reads the store
    // yet; see ruth/simulator/store_bridge.py.

    m.def("advance_store_reset",
        [](size_t n_vehicles, size_t n_route_nodes, size_t max_routing_id) {
            auto& store = ruthlib::VehicleStore::instance();
            store.clear();
            store.reserve(n_vehicles, n_route_nodes);
            store.resize_segments(max_routing_id);
        },
        py::arg("n_vehicles"), py::arg("n_route_nodes"), py::arg("max_routing_id"),
        "empty the store and size it for a fresh population");

    m.def("advance_store_set_segments",
        [](py::array_t<int64_t, py::array::c_style | py::array::forcecast> routing_id,
           py::array_t<double,  py::array::c_style | py::array::forcecast> length_m,
           py::array_t<double,  py::array::c_style | py::array::forcecast> max_speed_kph,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> lanes,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> is_closed) {
            const py::ssize_t n = routing_id.shape(0);
            if (length_m.shape(0) != n || max_speed_kph.shape(0) != n ||
                lanes.shape(0) != n || is_closed.shape(0) != n) {
                throw std::runtime_error("advance_store_set_segments: array lengths differ");
            }
            auto& store = ruthlib::VehicleStore::instance();
            const auto* rid = routing_id.data();
            const auto* len = length_m.data();
            const auto* kph = max_speed_kph.data();
            const auto* lan = lanes.data();
            const auto* cls = is_closed.data();
            for (py::ssize_t i = 0; i < n; ++i) {
                store.set_segment(rid[i], len[i], kph[i], lan[i], cls[i]);
            }
        },
        py::arg("routing_id"), py::arg("length_m"), py::arg("max_speed_kph"),
        py::arg("lanes"), py::arg("is_closed"),
        "bulk-fill the segment geometry table");

    // Vehicles arrive as columns plus a CSR of their routes, so the whole
    // population crosses the boundary in one call rather than one call each.
    m.def("advance_store_add_vehicles",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> vehicle_id,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> start_index,
           py::array_t<double,  py::array::c_style | py::array::forcecast> start_offset_m,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> time_offset_us,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> frequency_us,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> fcd_period_us,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> active,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> route_nodes,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> route_segment_ids,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> route_offsets) {
            const py::ssize_t n = vehicle_id.shape(0);
            if (route_offsets.shape(0) != n + 1) {
                throw std::runtime_error(
                    "advance_store_add_vehicles: route_offsets must have n+1 entries");
            }
            auto& store = ruthlib::VehicleStore::instance();
            const auto* off = route_offsets.data();
            const auto* nodes = route_nodes.data();
            const auto* segs = route_segment_ids.data();
            for (py::ssize_t i = 0; i < n; ++i) {
                const int32_t begin = off[i];
                const int32_t len = off[i + 1] - begin;
                store.add_vehicle(vehicle_id.at(i), start_index.at(i), start_offset_m.at(i),
                                  time_offset_us.at(i), frequency_us.at(i),
                                  fcd_period_us.at(i), active.at(i),
                                  nodes + begin, segs + begin, len);
            }
        },
        py::arg("vehicle_id"), py::arg("start_index"), py::arg("start_offset_m"),
        py::arg("time_offset_us"), py::arg("frequency_us"), py::arg("fcd_period_us"),
        py::arg("active"), py::arg("route_nodes"), py::arg("route_segment_ids"),
        py::arg("route_offsets"),
        "bulk-add vehicles from columns + a CSR of their routes");

    // Returned as raw bytes rather than a structured dtype declared here: the
    // Python side already owns the authoritative dtype (ADVANCE_TASK_DTYPE) and
    // views these bytes through it, so there is exactly one layout definition to
    // keep in step with the struct instead of two.
    m.def("advance_store_make_tasks",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> vehicle_ids) {
            auto& store = ruthlib::VehicleStore::instance();
            const py::ssize_t n = vehicle_ids.shape(0);
            auto* tasks = new std::vector<ruthlib::AdvanceTask>();
            tasks->reserve(static_cast<size_t>(n));
            for (py::ssize_t i = 0; i < n; ++i) {
                const int32_t slot = store.slot_of(vehicle_ids.at(i));
                if (slot < 0) {
                    delete tasks;
                    throw std::runtime_error(
                        "advance_store_make_tasks: vehicle " +
                        std::to_string(vehicle_ids.at(i)) + " is not in the store");
                }
                tasks->push_back(store.make_task(slot));
            }
            auto capsule = py::capsule(tasks, [](void* p) {
                delete reinterpret_cast<std::vector<ruthlib::AdvanceTask>*>(p);
            });
            const py::ssize_t n_bytes =
                static_cast<py::ssize_t>(tasks->size() * sizeof(ruthlib::AdvanceTask));
            return py::array_t<uint8_t>({n_bytes}, {sizeof(uint8_t)},
                                        reinterpret_cast<uint8_t*>(tasks->data()), capsule);
        },
        py::arg("vehicle_ids"),
        "AdvanceTasks built from the store, as raw bytes to view through "
        "ADVANCE_TASK_DTYPE");

    m.def("advance_store_size",
        []() { return ruthlib::VehicleStore::instance().size(); },
        "number of vehicles currently in the store");

    // --- step selection (shadow mode) ---------------------------------------

    m.def("advance_store_select_within_offset",
        [](int64_t offset_seconds, int64_t freq_seconds) {
            return vec_to_numpy(
                ruthlib::VehicleStore::instance().select_within_offset(
                    offset_seconds, freq_seconds));
        },
        py::arg("offset_seconds"), py::arg("freq_seconds"),
        "vehicle ids selected for this step, in slot order");

    // None rather than a sentinel, so the Python side compares against
    // compute_current_offset()'s None directly instead of inventing a rule for
    // what -1 means.
    m.def("advance_store_min_active_time_offset_us",
        []() -> py::object {
            int64_t out = 0;
            if (!ruthlib::VehicleStore::instance().min_active_time_offset_us(out)) {
                return py::none();
            }
            return py::int_(out);
        },
        "smallest non-zero time offset among active vehicles, or None");

    m.def("advance_store_count_active",
        []() { return ruthlib::VehicleStore::instance().count_active(); },
        "number of active vehicles in the store");

    m.def("advance_store_at_segment_end",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> vehicle_ids) {
            std::vector<int32_t> ids(vehicle_ids.data(),
                                     vehicle_ids.data() + vehicle_ids.shape(0));
            return vec_to_numpy(
                ruthlib::VehicleStore::instance().at_segment_end_ids(ids));
        },
        py::arg("vehicle_ids"),
        "of vehicle_ids, those sitting exactly at the end of their segment");
}
