import re
import h5py
import numpy as np
from collections import OrderedDict
from datetime import datetime

INT32_MAX = 2147483647


def coord_to_int(coord):
    return int(round(coord * pow(10, 6)))


def get_osmid_from_data(data):
    return data['osmid'][0] if type(data['osmid']) is list else data['osmid']


def get_speed_from_data(data):
    if 'speed_kph' in data:
        return data['speed_kph']

    speed_str = data.get('maxspeed', "50")
    speed_str = speed_str[0] if type(speed_str) is list else speed_str

    nums = re.findall(r'\d+', speed_str)
    r = int(nums[0]) if len(nums) > 0 else 50
    return r


def dict_to_dataset(dictionary, dtype):
    data_array = []

    for _, value in dictionary.items():
        data_array.append([value])

    return np.array(data_array, dtype=dtype)


def save_graph_to_hdf5(g, file_path):
    part_name = b'CZE'
    edge_index = 0
    node_type = [
        ("id", np.int32),
        ("latitudeInt", np.int32),
        ("longitudeInt", np.int32),
        ("edgeOutCount", np.uint8),
        ("edgeOutIndex", np.int32),
        ("edgeInCount", np.uint8)
    ]
    edge_type = [
        ("edgeId", np.int32),
        ("nodeIndex", np.int32),
        ("computed_speed", np.int32),
        ("length", np.int32),
        ("edgeDataIndex", np.int32)
    ]
    edge_data_type = [
        ('id', np.int32),
        ('speed', np.uint8),
        ('funcClass', np.uint8),
        ('lanes', np.uint8),
        ('vehicleAccess', np.uint8),
        ('specificInfo', np.uint16),
        ('maxWeight', np.uint16),
        ('maxHeight', np.uint16),
        ('maxAxleLoad', np.uint8),
        ('maxWidth', np.uint8),
        ('maxLength', np.uint8),
        ('incline', np.int8)
    ]
    node_map_type = [
        ('nodeId', np.int32),
        ('partId', np.dtype('S4')),
        ('nodeIndex', np.int32)
    ]

    nodes_array = []
    edges_array = []
    node_dict = OrderedDict()
    edge_data_dict = OrderedDict()

    # Dictionary to remap OSM node IDs to HDF map IDs
    # (to fit into the size of int32 used in the alternatives computation)
    osm_to_hdf_map_ids = {}

    for row_id, (node_id, data) in enumerate(g.nodes(data=True)):
        # Indexing with offset to avoid 0 index
        # The range of OSM node IDs is too wide for subtracting the min value
        osm_to_hdf_map_ids[node_id] = row_id + 1000
        assert row_id + 1000 not in node_dict
        node_dict[row_id + 1000] = (row_id + 1000, part_name, row_id)

    edge_data_index = 0
    for row_id, (id_from, id_to, edge_data) in enumerate(g.edges(data=True)):
        speed = get_speed_from_data(edge_data)
        func_class = 7
        lanes = edge_data.get('lanes', 1)
        lanes = int(lanes[0]) if type(lanes) is list else lanes
        vehicle_access = edge_data.get('vehicleAccess', 1)
        #         NoneVeh = 0,
        #         Regular = 1,
        #         PublicService = 2,
        #         Trucks = 4, // trucks above 3,5 t
        #         LightCommercialVehicles = 8, // trucks below 3,5 t
        #         LongerHeavierVehicle = 16, // trucks below 44 t
        specific_info = edge_data.get('specificInfo ', 0)
        #         None = 0,
        #         TollWay = 1,
        #         SlipRoad = 2,
        #         MultiCarriageway = 4,
        #         Roundabout = 8,
        #         BorderCross = 16,
        #         Bridge = 32,
        #         Tunnel = 64
        max_weight = edge_data.get('maxWeight', 65535)
        max_height = edge_data.get('maxHeight', 65535)
        max_axle_load = edge_data.get('maxAxleLoad', 255)
        max_width = edge_data.get('maxWidth', 255)
        max_length = edge_data.get('maxLength', 255)
        incline = edge_data.get('incline', 0)

        edge_data_tuple = (
            edge_data_index,
            speed,
            func_class,
            lanes,
            vehicle_access,
            specific_info,
            max_weight,
            max_height,
            max_axle_load,
            max_width,
            max_length,
            incline
        )

        osm_id = get_osmid_from_data(edge_data)
        if osm_id not in edge_data_dict:
            edge_data_dict[osm_id] = edge_data_tuple
            edge_data_index += 1

    for row_id, (node_id, node_data) in enumerate(g.nodes(data=True)):
        out_edges = g.out_edges(node_id, data=True)
        in_edges = g.in_edges(node_id)

        latitude_int = coord_to_int(node_data['x'])
        longitude_int = coord_to_int(node_data['y'])
        edge_out_count = len(out_edges)
        edge_in_count = len(in_edges)

        edge_out_index = edge_index
        edge_index += edge_out_count

        # set up edges from
        for id_from, id_to, edge_data in out_edges:
            edge_id = get_osmid_from_data(edge_data)
            node_index = node_dict[osm_to_hdf_map_ids[id_to]][2]
            computer_speed = get_speed_from_data(edge_data)
            length = edge_data['length']
            edge_data_index = edge_data_dict[edge_id][0]

            edge_tuple = (
                edge_id,
                node_index,
                computer_speed,
                length,
                edge_data_index
            )
            edges_array.append([edge_tuple])

        node_tuple = (
            osm_to_hdf_map_ids[node_id],
            latitude_int,
            longitude_int,
            edge_out_count,
            edge_out_index,
            edge_in_count
        )

        nodes_array.append([node_tuple])

    nodes_array = np.array(nodes_array, dtype=node_type)
    edges_array = np.array(edges_array, dtype=edge_type)
    edge_data_array = dict_to_dataset(edge_data_dict, dtype=edge_data_type)
    node_map_array = dict_to_dataset(node_dict, dtype=node_map_type)

    parts_info_type = [
        ("id", np.dtype('S4')),
        ("nodeCount", np.int64),
        ("edgeCount", np.int64),
    ]

    with h5py.File(file_path, 'w') as h:
        index_group = h.create_group("Index")
        index_group.attrs['PartsCount'] = 1
        index_group.attrs['CreationTime'] = np.datetime64(datetime.now()).astype('int64')
        index_group.attrs['PartsInfo'] = np.array([(part_name, len(nodes_array), len(edges_array))],
                                                  dtype=parts_info_type)

        country_group = index_group.create_group("CZE")
        country_group.attrs['PartInfo'] = part_name
        country_group.create_dataset("Nodes", data=nodes_array, compression="gzip", compression_opts=4)
        country_group.create_dataset("Edges", data=edges_array, compression="gzip", compression_opts=4)

        index_group.create_dataset("EdgeData", data=edge_data_array, compression="gzip", compression_opts=4)
        index_group.create_dataset("NodeMap", data=node_map_array, compression="gzip", compression_opts=4)

    return osm_to_hdf_map_ids
