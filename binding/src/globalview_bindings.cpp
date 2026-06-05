#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "globalview.hpp"

namespace py = pybind11;

/**
 * Python bindings for the C++ GlobalView implementation
 */
PYBIND11_MODULE(globalview_cpp, m) {
    m.doc() = "C++ implementation of GlobalView with spatial indexing";

    // Bind FCDRecord struct
    py::class_<FCDRecord>(m, "FCDRecord")
        .def(py::init<double, int, int64_t, double, double>(),
            py::arg("datetime_seconds"),
            py::arg("vehicle_id"),
            py::arg("segment_id"),
            py::arg("offset_from_start"),
            py::arg("vehicle_speed_mps"))
        .def_readwrite("datetime_seconds", &FCDRecord::datetime_seconds)
        .def_readwrite("vehicle_id", &FCDRecord::vehicle_id)
        .def_readwrite("segment_id", &FCDRecord::segment_id)
        .def_readwrite("offset_from_start", &FCDRecord::offset_from_start)
        .def_readwrite("vehicle_speed_mps", &FCDRecord::vehicle_speed_mps);

    // Bind GlobalView class
    py::class_<GlobalView, std::shared_ptr<GlobalView>>(m, "GlobalView")
        .def(py::init<>())
        .def("add", &GlobalView::add,
            py::arg("fcd"),
            "Add an FCD record to the global view")
        .def("add_batch", &GlobalView::add_batch,
            py::arg("fcds"),
            "Add multiple FCD records in batch (more efficient for bulk operations)")
        .def("number_of_vehicles_ahead",
            &GlobalView::number_of_vehicles_ahead,
            py::arg("datetime_seconds"),
            py::arg("segment_id"),
            py::arg("tolerance_seconds") = 0.0,
            py::arg("vehicle_id") = -1,
            py::arg("vehicle_offset_m") = 0.0,
            "Count vehicles ahead of a given position within a time window")
        .def("level_of_service_in_front_of_vehicle",
            &GlobalView::level_of_service_in_front_of_vehicle,
            py::arg("datetime_seconds"),
            py::arg("segment_id"),
            py::arg("segment_length"),
            py::arg("segment_lanes"),
            py::arg("tolerance_seconds") = 0.0,
            py::arg("vehicle_id") = -1,
            py::arg("vehicle_offset_m") = 0.0,
            "Calculate level of service in front of a vehicle")
        .def("level_of_service_in_time_at_segment",
            &GlobalView::level_of_service_in_time_at_segment,
            py::arg("datetime_seconds"),
            py::arg("segment_id"),
            py::arg("segment_length"),
            py::arg("segment_lanes"),
            py::arg("tolerance_seconds") = 0.0,
            "Calculate level of service for entire segment")
        .def("get_segment_speed",
            &GlobalView::get_segment_speed,
            py::arg("segment_id"),
            "Get average speed on a segment")
        .def("drop_old",
            &GlobalView::drop_old,
            py::arg("dt_threshold"),
            "Remove FCD records older than threshold, returns modified segments")
        .def("clear",
            &GlobalView::clear,
            "Clear all FCD records")
        .def("export_all_fcds",
            &GlobalView::export_all_fcds,
            "Export all FCD records for serialization (pickling)")
        .def("import_all_fcds",
            &GlobalView::import_all_fcds,
            py::arg("fcds"),
            "Import FCD records from serialization (unpickling)");
}
