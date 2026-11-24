class MPIDistributor:

    def __init__(self, threads=None, queue="prioritized"):
        # Number of ACE scheduler worker threads per rank, from the config.json
        # "ace" section (ace.threads). None / <= 0 falls back to all cores.
        self.threads = threads
        # ACE scheduler queue policy (ace.queue): "prioritized" or "fifo".
        self.queue = queue

    def __enter__(self):
        import ruthlib as ru
        self.ru = ru
        # Explicit ace.threads wins verbatim (ACE honors it exactly). Otherwise
        # pass 0 to signal "auto": ACE resizes best-effort from the available cores
        # and local ranks. (Previously this pre-filled cpu_count(), which ACE then
        # treated as an explicit request and re-divided.)
        num_threads = self.threads if self.threads and self.threads > 0 else 0
        ru.setup_ace(num_threads, self.queue)
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