import pytest
import glob
import os


@pytest.fixture(autouse=True)
def cleanup_h5_pickle():
    """Remove any .h5 and .pickle files in the current working directory before and after each test.

    This keeps tests isolated w.r.t. generated FCD HDF5 files and saved pickles. It operates only in
    the current working directory (does not recurse into subdirectories) to avoid removing repository assets.
    """
    os.makedirs("data", exist_ok=True)

    # pre-test cleanup
    for p in glob.glob("*.h5") + glob.glob("*.pickle"):
        try:
            os.remove(p)
        except Exception:
            pass

    yield

    # post-test cleanup
    for p in glob.glob("*.h5") + glob.glob("*.pickle"):
        try:
            os.remove(p)
        except Exception:
            pass



@pytest.fixture(scope="session")
def regression_vehicles_path(tmp_path_factory):
    """Small offline workload routed against the local test map, independent of benchmarks."""
    from datetime import timedelta
    from pathlib import Path
    import networkx as nx
    import pandas as pd
    from ruth.data.map import Map

    graphml = next((Path(__file__).parent / "data").glob("*.graphml"))
    routing_map = Map(graphml_file=str(graphml), save_hdf=False)
    origin, destination = 25664661, 305832258
    route = nx.shortest_path(routing_map.current_network, origin, destination,
                             weight="current_travel_time")
    north, west, south, east = routing_map.bbox.get_coords()
    rows = [dict(id=i + 1, origin_node=origin, dest_node=destination,
                 osm_route=route, time_offset=timedelta(seconds=(i + 1) * 5),
                 start_index=0, start_distance_offset=0.0, active=True,
                 status="not started", frequency=timedelta(seconds=5),
                 fcd_sampling_period=timedelta(seconds=5),
                 download_date=routing_map.download_date,
                 bbox_lat_max=north, bbox_lon_min=west,
                 bbox_lat_min=south, bbox_lon_max=east)
            for i in range(10)]
    path = tmp_path_factory.mktemp("regression-input") / "vehicles.parquet"
    pd.DataFrame(rows).to_parquet(path, engine="fastparquet", object_encoding="json", index=False)
    return str(path)
