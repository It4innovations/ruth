from collections import defaultdict, deque
from ruth.vehicle import Vehicle


class QueuesManager:
    queues = defaultdict(deque)

    @staticmethod
    def add_to_queue(vehicle: Vehicle):
        QueuesManager.queues[(vehicle.current_node, vehicle.next_node)].append(vehicle)

    @staticmethod
    def remove_vehicle(vehicle: Vehicle, node_from, node_to):
        popped_vehicle = QueuesManager.queues[(node_from, node_to)].popleft()
        assert popped_vehicle == vehicle
