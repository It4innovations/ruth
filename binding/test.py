import time
import tracemalloc
import logging
import argparse
import os
import pickle
import multiprocessing

import ruthlib as ru

# Default values
DEFAULT_NUM_VEHICLES = 10_000
DEFAULT_DATA_DIR = "dataset"
DEFAULT_MAP_PATH = "map_prague.hdf5"
DEFAULT_VEHICLE_PATH = "OD_matrix.csv"


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process vehicle routing data')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Directory containing input data (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--map-path', type=str, default=DEFAULT_MAP_PATH,
                        help=f'Path to the map file (default: {DEFAULT_MAP_PATH})')
    parser.add_argument('--vehicle-path', type=str, default=DEFAULT_VEHICLE_PATH,
                        help=f'Path to the vehicle data file (default: {DEFAULT_VEHICLE_PATH})')
    parser.add_argument('--num-vehicles', type=int, default=DEFAULT_NUM_VEHICLES,
                        help=f'Number of vehicles to process (default: {DEFAULT_NUM_VEHICLES})')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')
    return parser.parse_args()

def setup_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('routing.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_level)
    
    num_threads = multiprocessing.cpu_count() # Use all available CPU cores
    logger.info(f"Using {num_threads} threads")
    

    # Setup ACE with number of threads
    ru.setup_ace(num_threads)
    
    # Construct full file paths
    map_path = os.path.join(args.data_dir, args.map_path)
    vehicle_path = os.path.join(args.data_dir, args.vehicle_path)
    
    logger.info(f"Using data directory: {os.path.abspath(args.data_dir)}")
    logger.info(f"Processing up to {args.num_vehicles} vehicles")
    role = "Master" if ru.is_master() else "Worker"
    logger.info(f"Running as {role}")

    # tracemalloc.start()
    ru.init_routes()

    if ru.is_master():
        OD_matrix = ru.load_od_matrix(vehicle_path, 10)
        ru.setup_map(map_path)

        try:
            logger.info("Starting alternatives computation...")
            start_time = time.perf_counter()
            # start_snapshot = tracemalloc.take_snapshot()

            ru.do_alternatives(OD_matrix, 2)

            logger.info(f"Alternatives computation finished in {time.perf_counter() - start_time:.2f} seconds.")
            vehicle_ids, routes_per_vehicle, travel_times_per_vehicle = ru.get_routes()
            logger.info(f"Plus Routes retrieved in {time.perf_counter() - start_time:.2f} seconds.")

            end_time = time.perf_counter()
            # end_snapshot = tracemalloc.take_snapshot()

            # calc travel time
            routes_per_one_vehicle = routes_per_vehicle[0]
            edge_speeds = [(1, 0.0)]

            ru.update_speeds(edge_speeds)

            start_time_tt = time.perf_counter()
            ru.do_travel_times(routes_per_one_vehicle)

            travel_times = ru.get_travel_times()
            logger.info(f"{len(travel_times)} travel times computed in {time.perf_counter() - start_time_tt:.2f} seconds.")

        except Exception as e:
            logger.error(f"Error during alternatives computation: {str(e)}")
            raise

        # Calculate time and memory differences
        duration_ms = (end_time - start_time) * 1000
        logger.info(f"Alternatives duration: {duration_ms:10.2f} ms")




    # ru.barrier()

    ru.finalize()

if __name__ == "__main__":
    main()
