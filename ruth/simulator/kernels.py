from typing import List

from ..vehicle import Vehicle


class KernelProvider:
    """
    Provides implementation of computationally intensive tasks.
    """
    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[List[List[int]]]:
        raise NotImplementedError


class LocalKernelProvider(KernelProvider):
    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[List[List[int]]]:
        results = []
        for vehicle in vehicles:
            results.append(vehicle.k_shortest_paths(k))
        return results
