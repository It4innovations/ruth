#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "globalview.hpp"

namespace py = pybind11;

using FCDArray = py::array_t<FCDRecord, py::array::c_style | py::array::forcecast>;

/**
 * Python bindings for the C++ GlobalView implementation
 */
PYBIND11_MODULE(globalview_cpp, m) {
    m.doc() = "C++ implementation of GlobalView with spatial indexing";

    // Structured-array view of FCDRecord: a NumPy array with this dtype maps
    // byte-for-byte onto the POD, so batches cross the boundary without
    // creating per-record Python objects.
    PYBIND11_NUMPY_DTYPE(FCDRecord, datetime_seconds, vehicle_id, segment_id,
                         offset_from_start, vehicle_speed_mps);

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
        .def("add_batch_array",
            [](GlobalView& self, const FCDArray& fcds) {
                auto buf = fcds.request();
                const auto* data = static_cast<const FCDRecord*>(buf.ptr);
                const auto n = static_cast<size_t>(buf.size);
                py::gil_scoped_release release;
                self.add_batch_raw(data, n);
            },
            py::arg("fcds"),
            "Add records from a NumPy structured array (fcd_dtype); single copy, GIL released")
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
            py::call_guard<py::gil_scoped_release>(),
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
            "Import FCD records from serialization (unpickling)")
        .def("export_all_fcds_array",
            [](const GlobalView& self) {
                const size_t n = self.total_fcds();
                py::array_t<FCDRecord> arr(static_cast<py::ssize_t>(n));
                auto* out = static_cast<FCDRecord*>(arr.request().ptr);
                py::gil_scoped_release release;
                self.export_all_fcds_into(out);
                return arr;
            },
            "Export all FCD records as a NumPy structured array (fcd_dtype); single copy")
        .def("import_all_fcds_array",
            [](GlobalView& self, const FCDArray& fcds) {
                auto buf = fcds.request();
                const auto* data = static_cast<const FCDRecord*>(buf.ptr);
                const auto n = static_cast<size_t>(buf.size);
                py::gil_scoped_release release;
                self.import_all_fcds_raw(data, n);
            },
            py::arg("fcds"),
            "Import records from a NumPy structured array (fcd_dtype); single copy, GIL released");

    m.attr("fcd_dtype") = py::dtype::of<FCDRecord>();
}
