from collections import defaultdict, deque
from ..vehicle import Vehicle


class QueuesManager:

    def __init__(self):
        self.queues = defaultdict(deque)
        self.to_be_added = list()
        self.to_be_removed = list()

    def add_to_queue(self, vehicle: Vehicle):
        self.to_be_added.append((vehicle.id, vehicle.current_node, vehicle.next_node))

    def remove_inactive_vehicle(self, vehicle: Vehicle):
        # check if vehicle is in to_be_removed
        for i, (vehicle_id, node_from, node_to) in enumerate(self.to_be_removed):
            if vehicle_id == vehicle.id:
                return
        # check if vehicle is in any queue
        for (node_from, node_to), queue in self.queues.items():
            if vehicle.id in queue:
                raise ValueError("Vehicle not found in to_be_removed")
        return


    def batch_update(self):
        for vehicle_id, node_from, node_to in self.to_be_removed:
            queue = self.queues[(node_from, node_to)]
            assert len(queue) != 0
            popped_vehicle_id = queue.popleft()
            assert popped_vehicle_id == vehicle_id

        self.to_be_removed.clear()

        for vehicle_id, node_from, node_to in self.to_be_added:
            queue = self.queues[(node_from, node_to)]
            queue.append(vehicle_id)

        self.to_be_added.clear()

    def remove_vehicle(self, vehicle: Vehicle, node_from, node_to):
        self.to_be_removed.append((vehicle.id, node_from, node_to))
