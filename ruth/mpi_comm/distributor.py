import multiprocessing

class MPIDistributor:

    def __enter__(self):
        import ruthlib as ru
        self.ru = ru
        num_threads = multiprocessing.cpu_count()  # Use all available CPU cores
        ru.setup_ace(num_threads)
        ru.init_routes()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.ru.finalize()

    @staticmethod
    def allow_mpi() -> bool:
        try:
            import ruthlib as ru
            return True
        except ImportError:
            return False

    @staticmethod
    def is_master() -> bool:
        try:
            import ruthlib as ru
            return ru.is_master()
        except ImportError:
            return True

def main():
    with MPIDistributor() as distributor:
        if distributor.is_master():
           print("Master process is setting up the map and initializing routes.")
        else:
           print("Worker process is waiting for tasks.")

        print("Program finished")

if __name__ == "__main__":
    main()