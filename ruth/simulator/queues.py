from collections import defaultdict, deque

from ruth.vehicle import Vehicle


class QueuesManager:
    queues = defaultdict(deque)

    @staticmethod
    def filter_cars(vehicles: [Vehicle]):
        filtered_vehicles = []
        waiting_vehicles = []
        for vehicle in vehicles:
            current_queue = QueuesManager.queues[(vehicle.current_node, vehicle.next_node)]
            if vehicle not in current_queue:
                filtered_vehicles.append(vehicle)
                continue

            first_vehicle = current_queue[0]
            if vehicle == first_vehicle:
                filtered_vehicles.append(vehicle)
            elif first_vehicle.time_offset != vehicle.time_offset:
                waiting_vehicles.append(vehicle)

        return filtered_vehicles, waiting_vehicles

    @staticmethod
    def add_to_queue(vehicle: Vehicle):
        QueuesManager.queues[(vehicle.current_node, vehicle.next_node)].append(vehicle)

    @staticmethod
    def remove_vehicle(vehicle: Vehicle, node_from, node_to):
        popped_vehicle = QueuesManager.queues[(node_from, node_to)].popleft()
        assert popped_vehicle == vehicle
