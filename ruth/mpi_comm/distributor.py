import ruthlib as ru
import multiprocessing

class MPIDistributor:

    def __enter__(self):
        num_threads = multiprocessing.cpu_count()  # Use all available CPU cores
        ru.setup_ace(num_threads)
        ru.init_routes()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ru.finalize()

    @staticmethod
    def is_master() -> bool:
        return ru.is_master()

    @staticmethod
    def barrier():
        ru.barrier()

    @staticmethod
    def is_simulation_running() -> bool:
        return ru.is_simulation_running()

    @staticmethod
    def finish_simulation():
        if ru.is_master():
            ru.finish_simulation()
        else:
            print("Worker process cannot finish simulation.")

def main():
    with MPIDistributor() as distributor:
        if distributor.is_master():
           print("Master process is setting up the map and initializing routes.")
        else:
           print("Worker process is waiting for tasks.")

        print("Program finished")

if __name__ == "__main__":
    main()