import sys
from datetime import datetime, timedelta


# # Add the C++ module location to sys.path BEFORE importing
sys.path.insert(0, '/home/pav/Repos/ruth-opencode/binding/build/lib')
sys.path.insert(0, '/home/pav/Repos/ruth-opencode')


from ruth.globalview import GlobalView as PythonGlobalView
from ruth.globalview_cpp import GlobalView as CppGlobalView


def test_verify_cpp_is_being_used():
    try:
        import globalview_cpp
        print("C++ module (globalview_cpp) is available")
    except ImportError:
        raise RuntimeError("C++ GlobalView module not found! Test requires C++ implementation.")
    print("CppGlobalView is configured to use C++ implementation")

# Mock classes for testing
class MockSegment:
    def __init__(self, segment_id, length, lanes):
        self.id = segment_id
        self.length = length
        self.lanes = lanes

class MockFCDRecord:
    def __init__(self, datetime, vehicle_id, segment, offset, speed, status='A', active=True):
        self.datetime = datetime
        self.vehicle_id = vehicle_id
        self.segment = segment
        self.offset_from_start = offset
        self.vehicle_speed_mps = speed
        self.status = status
        self.active = active


def test_number_of_vehicles_ahead():
    """Test counting vehicles ahead on a segment"""
    print("Testing number_of_vehicles_ahead...")

    py_view = PythonGlobalView()
    cpp_view = CppGlobalView()

    segment = MockSegment(1, 1000, 2)
    base_time = datetime(2025, 1, 1, 12, 0, 0)

    # Add some FCD records
    for i in range(5):
        for offset in [100 + i*50, 300 + i*50, 500 + i*50]:
            fcd = MockFCDRecord(
                datetime=base_time + timedelta(seconds=i),
                vehicle_id=i,
                segment=segment,
                offset=offset,
                speed=20.0 + i,
                status='A',
                active=True
            )
            py_view.add(fcd)
            cpp_view.add(fcd)

    # Test counting vehicles ahead
    test_cases = [
        (200, -1, 0),      # 0 vehicles ahead of 200m
        (300, -1, 1),      # Some vehicles ahead of 300m
        (400, -1, 2),      # More vehicles
        (450, 0, 3),       # Exclude vehicle 0
    ]

    for offset, exclude_id, idx in test_cases:
        tolerance = timedelta(seconds=5)
        query_time = base_time + timedelta(seconds=idx)

        py_result = py_view.number_of_vehicles_ahead(
            query_time, segment.id, tolerance, exclude_id, offset
        )
        cpp_result = cpp_view.number_of_vehicles_ahead(
            query_time, segment.id, tolerance, exclude_id, offset
        )

        print(f"  Offset={offset}m, exclude={exclude_id}, time_idx={idx}: "
              f"Python={py_result}, C++={cpp_result}")

        assert py_result == cpp_result, \
            f"Mismatch at offset={offset}: Python={py_result}, C++={cpp_result}"


def test_level_of_service():
    """Test level of service calculation"""
    print("\nTesting level_of_service_in_front_of_vehicle...")

    py_view = PythonGlobalView()
    cpp_view = CppGlobalView()

    segment = MockSegment(42, 2000, 3)
    base_time = datetime(2025, 1, 1, 12, 0, 0)

    # Add many FCD records on the segment
    for i in range(20):
        for j in range(3):
            fcd = MockFCDRecord(
                datetime=base_time,
                vehicle_id=i*3 + j,
                segment=segment,
                offset=100 + i*100,
                speed=15.0 + j*2,
                status='A',
                active=True
            )
            py_view.add(fcd)
            cpp_view.add(fcd)

    # Test LoS calculation
    query_offsets = [0, 500, 1000, 1500]

    for offset in query_offsets:
        py_los = py_view.level_of_service_in_front_of_vehicle(
            base_time, segment, -1, offset, timedelta(seconds=0)
        )
        cpp_los = cpp_view.level_of_service_in_front_of_vehicle(
            base_time, segment, -1, offset, timedelta(seconds=0)
        )

        diff = abs(py_los - cpp_los)
        print(f"  Offset={offset}m: Python={py_los:.4f}, C++={cpp_los:.4f}, diff={diff:.6f}")

        # Allow small floating point differences
        assert diff < 1e-5, f"LoS mismatch at offset={offset}: diff={diff}"


def test_segment_speed():
    """Test average segment speed calculation"""
    print("\nTesting get_segment_speed...")

    py_view = PythonGlobalView()
    cpp_view = CppGlobalView()

    segment = MockSegment(99, 1500, 2)
    base_time = datetime(2025, 1, 1, 12, 0, 0)

    # Add FCD records with known speeds
    speeds = [10.0, 15.0, 20.0, 25.0, 30.0]
    for i, speed in enumerate(speeds):
        fcd = MockFCDRecord(
            datetime=base_time,
            vehicle_id=i,
            segment=segment,
            offset=100 + i*50,
            speed=speed,
            status='A',
            active=True
        )
        py_view.add(fcd)
        cpp_view.add(fcd)

    py_speed = py_view.get_segment_speed(segment.id)
    cpp_speed = cpp_view.get_segment_speed(segment.id)

    # Expected speed: average of speeds array in m/s
    # (Note: SpeedKph is a misnomer - it actually returns m/s in the original code)
    expected_avg_mps = sum(speeds) / len(speeds)

    print(f"  Expected: {expected_avg_mps:.2f} m/s")
    print(f"  Python:   {py_speed:.2f} m/s")
    print(f"  C++:      {cpp_speed:.2f} m/s")

    assert py_speed is not None and cpp_speed is not None
    assert abs(py_speed - cpp_speed) < 0.01, \
        f"Speed mismatch: Python={py_speed}, C++={cpp_speed}"


def test_drop_old():
    """Test dropping old FCD records"""
    print("\nTesting drop_old...")

    py_view = PythonGlobalView()
    cpp_view = CppGlobalView()

    segment = MockSegment(5, 500, 1)
    base_time = datetime(2025, 1, 1, 12, 0, 0)

    # Add records at different times
    for i in range(10):
        for j in range(3):
            fcd = MockFCDRecord(
                datetime=base_time + timedelta(minutes=i),
                vehicle_id=i*3 + j,
                segment=segment,
                offset=50 + i*10,
                speed=20.0,
                status='A',
                active=True
            )
            py_view.add(fcd)
            cpp_view.add(fcd)

    # Drop records older than 5 minutes
    threshold = base_time + timedelta(minutes=5)
    py_view.drop_old(threshold)
    cpp_view.drop_old(threshold)

    # TODO: Verify both have same count after dropping


def test_empty_segment():
    """Test handling of empty segments"""
    print("\nTesting empty segment handling...")

    py_view = PythonGlobalView()
    cpp_view = CppGlobalView()

    base_time = datetime(2025, 1, 1, 12, 0, 0)

    count_py = py_view.number_of_vehicles_ahead(base_time, 999, timedelta(seconds=0), -1, 0)
    count_cpp = cpp_view.number_of_vehicles_ahead(base_time, 999, timedelta(seconds=0), -1, 0)

    assert count_py == count_cpp == 0, "Empty segment should return 0"
    print("  Empty segment correctly returns 0")


def run_all_tests():
    test_verify_cpp_is_being_used()
    test_number_of_vehicles_ahead()
    test_level_of_service()
    test_segment_speed()
    test_drop_old()
    test_empty_segment()


if __name__ == "__main__":
    run_all_tests()
