from collections import defaultdict, deque
from ..vehicle import Vehicle


class QueuesManager:

    def __init__(self):
        self.queues = defaultdict(deque)

    def add_to_queue(self, vehicle: Vehicle):
        self.queues[(vehicle.current_node, vehicle.next_node)].append(vehicle.id)

    def remove_inactive_vehicle(self, vehicle: Vehicle):
        for queue in self.queues.values():
            if vehicle.id in queue:
                queue.remove(vehicle.id)

    def remove_vehicle(self, vehicle: Vehicle, node_from, node_to):
        queue = self.queues[(node_from, node_to)]
        assert len(queue) != 0
        popped_vehicle_id = queue.popleft()
        assert popped_vehicle_id == vehicle.id
