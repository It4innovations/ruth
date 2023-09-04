from collections import defaultdict, deque
from ..vehicle import Vehicle


class QueuesManager:

    def __init__(self):
        self.queues = defaultdict(deque)

    def add_to_queue(self, vehicle: Vehicle):
        self.queues[(vehicle.current_node, vehicle.next_node)].append(vehicle)

    def remove_vehicle(self, vehicle: Vehicle, node_from, node_to):
        queue = self.queues[(node_from, node_to)]
        if len(queue) != 0:
            check = queue[0]
            if check == vehicle:
                popped_vehicle = queue.popleft()
                # assert popped_vehicle == vehicle
