#!/usr/bin/env python3
"""
Production Workload Benchmark: Python Linear vs Python Indexed vs C++ GlobalView

Separates add_batch() initialization overhead from normal operation by calling multiple times.
Shows first call vs average of subsequent calls to isolate one-time costs.

Realistic simulation scenarios:
- Baseline: 10,000 FCD records per call × 5 calls (demonstrating init vs normal)
          - 10,000 level_of_service_in_front_of_vehicle calls
          - 1,000 get_segment_speed calls
          - 1 drop_old call
          - 50 segments (~200 FCD/segment)

- Scaled: Same but with 50,000 queries and ~200 segments (~50 FCD/segment)
"""
import sys
import time
from datetime import datetime, timedelta
import logging
import glob
import os

logging.getLogger().setLevel(logging.CRITICAL)

from ruth.data.map import Map
from ruth.globalview import GlobalView as PythonGlobalView
from ruth.globalview_indexed import GlobalViewIndexed as PythonGlobalViewIndexed

class MockFCDRecord:
    def __init__(self, dt, vehicle_id, segment, offset, speed, status='A', active=True):
        self.datetime = dt
        self.vehicle_id = vehicle_id
        self.segment = segment
        self.offset_from_start = offset
        self.vehicle_speed_mps = speed
        self.status = status
        self.active = active

base_time = datetime(2024, 1, 1, 12, 0, 0)
NUM_ADD_CALLS = 5  # Call add_batch 5 times to separate init from normal

# Load real map - use baseline (smallest) for baseline tests
graphml_files = sorted(glob.glob("./data/*.graphml"))
baseline_graphml = graphml_files[0]

# Load a larger map for scaled tests
scaled_graphml = graphml_files[2] if len(graphml_files) > 2 else graphml_files[-1]

print(f"Baseline map: {os.path.basename(baseline_graphml)}")
print(f"Scaled map: {os.path.basename(scaled_graphml)}")
print(f"Note: Each implementation will call add_batch() {NUM_ADD_CALLS} times to show initialization overhead separately\n")

routing_map = Map(graphml_file=baseline_graphml, data_dir="./data", with_speeds=True, save_hdf=False)
routing_map_scaled = Map(graphml_file=scaled_graphml, data_dir="./data", with_speeds=True, save_hdf=False)

# Load segments
segments_list = []
for routing_id in range(1, min(51, len(routing_map.routing_id_to_node_ids) + 1)):
    try:
        segment_id_tuple = routing_map.int_to_segment_id(routing_id)
        node_from, node_to = segment_id_tuple
        segment = routing_map.get_osm_segment(node_from, node_to)
        segments_list.append((segment_id_tuple, segment))
    except:
        continue

segments_list_scaled = []
for routing_id in range(1, min(201, len(routing_map_scaled.routing_id_to_node_ids) + 1)):
    try:
        segment_id_tuple = routing_map_scaled.int_to_segment_id(routing_id)
        node_from, node_to = segment_id_tuple
        segment = routing_map_scaled.get_osm_segment(node_from, node_to)
        segments_list_scaled.append((segment_id_tuple, segment))
    except:
        continue

# Load C++ module first with error handling
sys.path.insert(0, './binding/build/lib')
try:
    from ruth.globalview_wrapper import GlobalView as CppGlobalView
    CPP_AVAILABLE = True
    print("✓ C++ GlobalView module loaded successfully\n")
except ImportError as e:
    CPP_AVAILABLE = False
    print(f"✗ FAILED to load C++ GlobalView module: {e}\n")
    sys.exit(1)

def benchmark_scenario(scenario_name, routing_map, segments_list, num_add_calls=5, num_los_calls=10000, num_speed_calls=1000):
    """Run complete benchmark scenario with multiple add_batch calls"""
    fcd_per_segment = 10000 / len(segments_list)

    print("=" * 150)
    print(f"{scenario_name.upper()}")
    print("=" * 150)
    print(f"Segments: {len(segments_list)} | FCD per segment: ~{fcd_per_segment:.1f}")
    print(f"Queries: level_of_service={num_los_calls:,} | get_segment_speed={num_speed_calls:,}")
    print()

    # Pre-generate FCD records to add
    fcds = []
    for i in range(10000):
        segment_id_tuple, segment = segments_list[i % len(segments_list)]
        fcd = MockFCDRecord(
            base_time + timedelta(seconds=i * 0.1),
            i, segment,
            100.0 + (i % 500) * 2.0,
            15.0 + (i % 20)
        )
        fcds.append(fcd)

    # ========== PYTHON LINEAR ==========
    py_linear_gv = PythonGlobalView(routing_map=routing_map)
    py_linear_add_times = []
    print("Python Linear - add_batch times:")
    for call_num in range(num_add_calls):
        start = time.perf_counter()
        for fcd in fcds:
            py_linear_gv.add(fcd)
        elapsed = (time.perf_counter() - start) * 1000
        py_linear_add_times.append(elapsed)
        print(f"  Call {call_num+1}: {elapsed:.2f} ms")

    # Queries on Python Linear
    start = time.perf_counter()
    for i in range(num_los_calls):
        segment_id_tuple, segment = segments_list[i % len(segments_list)]
        los = py_linear_gv.level_of_service_in_front_of_vehicle(
            base_time + timedelta(seconds=5), segment, vehicle_id=10,
            vehicle_offset_m=200.0, tolerance=timedelta(seconds=10)
        )
    py_linear_los = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    for i in range(num_speed_calls):
        segment_id_tuple, segment = segments_list[i % len(segments_list)]
        speed = py_linear_gv.get_segment_speed(segment_id_tuple)
    py_linear_speed = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    py_linear_gv.drop_old(base_time - timedelta(hours=1))
    py_linear_drop = (time.perf_counter() - start) * 1000

    # ========== PYTHON INDEXED ==========
    py_indexed_gv = PythonGlobalViewIndexed(routing_map=routing_map)
    py_indexed_add_times = []
    print("\nPython Indexed - add_batch times:")
    for call_num in range(num_add_calls):
        start = time.perf_counter()
        py_indexed_gv.add_batch(fcds)
        elapsed = (time.perf_counter() - start) * 1000
        py_indexed_add_times.append(elapsed)
        print(f"  Call {call_num+1}: {elapsed:.2f} ms")

    # Queries on Python Indexed
    start = time.perf_counter()
    for i in range(num_los_calls):
        segment_id_tuple, segment = segments_list[i % len(segments_list)]
        los = py_indexed_gv.level_of_service_in_front_of_vehicle(
            base_time + timedelta(seconds=5), segment, vehicle_id=10,
            vehicle_offset_m=200.0, tolerance=timedelta(seconds=10)
        )
    py_indexed_los = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    for i in range(num_speed_calls):
        segment_id_tuple, segment = segments_list[i % len(segments_list)]
        speed = py_indexed_gv.get_segment_speed(segment_id_tuple)
    py_indexed_speed = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    py_indexed_gv.drop_old(base_time - timedelta(hours=1))
    py_indexed_drop = (time.perf_counter() - start) * 1000

    # ========== C++ ==========
    cpp_gv = CppGlobalView(routing_map=routing_map)
    cpp_add_times = []
    print("\nC++ - add_batch times:")
    for call_num in range(num_add_calls):
        start = time.perf_counter()
        cpp_gv.add_batch(fcds)
        elapsed = (time.perf_counter() - start) * 1000
        cpp_add_times.append(elapsed)
        print(f"  Call {call_num+1}: {elapsed:.2f} ms")

    # Queries on C++
    start = time.perf_counter()
    for i in range(num_los_calls):
        segment_id_tuple, segment = segments_list[i % len(segments_list)]
        los = cpp_gv.level_of_service_in_front_of_vehicle(
            base_time + timedelta(seconds=5), segment, vehicle_id=10,
            vehicle_offset_m=200.0, tolerance=timedelta(seconds=10)
        )
    cpp_los = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    for i in range(num_speed_calls):
        segment_id_tuple, segment = segments_list[i % len(segments_list)]
        speed = cpp_gv.get_segment_speed(segment_id_tuple)
    cpp_speed = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    cpp_gv.drop_old(base_time - timedelta(hours=1))
    cpp_drop = (time.perf_counter() - start) * 1000

    # ========== SUMMARY TABLE ==========
    print("\n" + "=" * 150)
    print("ADD_BATCH SUMMARY")
    print("=" * 150)
    print(f"{'Implementation':<20} {'1st Call (ms)':<18} {'Avg Calls 2-5 (ms)':<20} {'Init Overhead (ms)':<20}")
    print("-" * 150)

    py_linear_avg = sum(py_linear_add_times[1:]) / (len(py_linear_add_times) - 1) if len(py_linear_add_times) > 1 else py_linear_add_times[0]
    py_linear_init = py_linear_add_times[0] - py_linear_avg
    print(f"{'Python Linear':<20} {py_linear_add_times[0]:<18.2f} {py_linear_avg:<20.2f} {py_linear_init:<20.2f}")

    py_indexed_avg = sum(py_indexed_add_times[1:]) / (len(py_indexed_add_times) - 1) if len(py_indexed_add_times) > 1 else py_indexed_add_times[0]
    py_indexed_init = py_indexed_add_times[0] - py_indexed_avg
    print(f"{'Python Indexed':<20} {py_indexed_add_times[0]:<18.2f} {py_indexed_avg:<20.2f} {py_indexed_init:<20.2f}")

    cpp_avg = sum(cpp_add_times[1:]) / (len(cpp_add_times) - 1) if len(cpp_add_times) > 1 else cpp_add_times[0]
    cpp_init = cpp_add_times[0] - cpp_avg
    print(f"{'C++':<20} {cpp_add_times[0]:<18.2f} {cpp_avg:<20.2f} {cpp_init:<20.2f}")

    print("\n" + "=" * 150)
    print("QUERY & MAINTENANCE OPERATIONS")
    print("=" * 150)
    print(f"{'Operation':<50} {'Python Linear (ms)':<22} {'Python Indexed (ms)':<22} {'C++ (ms)':<20}")
    print("-" * 150)
    print(f"{'level_of_service_in_front_of_vehicle (' + str(num_los_calls) + ')':<50} {py_linear_los:<22.2f} {py_indexed_los:<22.2f} {cpp_los:<20.2f}")
    print(f"{'get_segment_speed (' + str(num_speed_calls) + ')':<50} {py_linear_speed:<22.2f} {py_indexed_speed:<22.2f} {cpp_speed:<20.2f}")
    print(f"{'drop_old (1)':<50} {py_linear_drop:<22.2f} {py_indexed_drop:<22.2f} {cpp_drop:<20.2f}")
    print("-" * 150)
    total_py_linear = sum(py_linear_add_times) + py_linear_los + py_linear_speed + py_linear_drop
    total_py_indexed = sum(py_indexed_add_times) + py_indexed_los + py_indexed_speed + py_indexed_drop
    total_cpp = sum(cpp_add_times) + cpp_los + cpp_speed + cpp_drop
    print(f"{'TOTAL (5 add_batch calls + queries)':<50} {total_py_linear:<22.2f} {total_py_indexed:<22.2f} {total_cpp:<20.2f}")
    print("=" * 150)

    print("\n" + "=" * 80)
    print("SPEEDUP (C++ vs Python Linear for queries only)")
    print("=" * 80)
    print(f"level_of_service_in_front_of_vehicle: {py_linear_los/cpp_los:.2f}x")
    print(f"get_segment_speed: {py_linear_speed/cpp_speed:.2f}x")
    print(f"drop_old: {py_linear_drop/cpp_drop:.2f}x")
    print("=" * 80 + "\n")

    return {
        'scenario': scenario_name,
        'py_linear_add': py_linear_add_times,
        'py_indexed_add': py_indexed_add_times,
        'cpp_add': cpp_add_times,
        'py_linear_los': py_linear_los,
        'py_indexed_los': py_indexed_los,
        'cpp_los': cpp_los,
        'py_linear_speed': py_linear_speed,
        'py_indexed_speed': py_indexed_speed,
        'cpp_speed': cpp_speed,
        'py_linear_drop': py_linear_drop,
        'py_indexed_drop': py_indexed_drop,
        'cpp_drop': cpp_drop,
        'total_py_linear': total_py_linear,
        'total_py_indexed': total_py_indexed,
        'total_cpp': total_cpp,
    }

# Run benchmarks
baseline_results = benchmark_scenario("Baseline (10,000 queries)", routing_map, segments_list, num_add_calls=NUM_ADD_CALLS, num_los_calls=10000, num_speed_calls=1000)
print("\n" + "█" * 150)
print("█" * 150 + "\n")
scaled_results = benchmark_scenario("Scaled (50,000 queries)", routing_map_scaled, segments_list_scaled, num_add_calls=NUM_ADD_CALLS, num_los_calls=50000, num_speed_calls=5000)