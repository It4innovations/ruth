import csv
from shapely.geometry import LineString, Point
import osmnx as ox
import datetime

from ruth.simulator import Simulation

### TO EXECUTE, PLEASE FILL THE FOLLOWING VARIABLES:
## pickle_file_path, graph_file and fcd_folder_path


# path to the output of the simulator (pickle file)
## fill with the path, such as "/YOUR_DIRECTORY/simulation_record_1.pickle"
pickle_file_path = ""

# path to the graphml that was used in the simulation
## fill with the path to graphml, such as "/YOUR_DIRECTORY/fa69dfba38a1a83d4fe27f3f8ef0b45b_town.graphml"
graph_file = ""

# path to the directory where the csv file is to be saved after generation and later loaded
## fill with the working path, such as "/YOUR_DIRECTORY/"
fcd_folder_path = ""

# name of the csv file
## fill with the output name, such as "simulation_record_1"
fcd_file = "simulation_record_1"

# suffix for the output of this script
ouput_file_possix = "_GPS"

### converting from pickle to CSV
print('Loading pickle file ...')
sim = Simulation.load(pickle_file_path)

print('Loading FCD data ...')
history_df = sim.history.to_dataframe()

print('Converting and saving to CSV file  ...')
history_df.to_csv(fcd_folder_path + fcd_file + '.csv', index=False)

# Load graphml file
print('Loading Graphml ...')
G = ox.load_graphml(filepath=graph_file, graphml_str=None, node_dtypes=None, edge_dtypes=None, graph_dtypes=None)
ct = datetime.datetime.now()

# Load fcd file
print('Loading FCD data from CSV file ...')
edges = []
with open(fcd_folder_path + fcd_file + '.csv', 'r') as edges_file:
    csv_reader = csv.reader(edges_file)
    next(csv_reader)
    for row in csv_reader:
        edges.append(row)

# Create date from first row
datum = ''
for edge in edges:
    timestamp = edge[0]
    datum, time = timestamp.split(" ")
    datum = datetime.datetime.strptime(datum, "%Y-%m-%d").strftime("%y%m%d")
    break

# Load nodes from graph and create dictionary
nodes_dict = {}
for node_id, node_data in G.nodes(data=True):
    geom_point = Point(node_data["x"], node_data["y"])
    nodes_dict[node_id] = geom_point

# Load geom from DiGraph and create dictionary for FROM, TO = GEOM
linestrings_dict = {}
for u, v, k, data in G.edges(keys=True, data=True):

    # if the geom is present then is used, otherwise it has to be created from points
    # OSMnx only adds a geometry attribute to simplified edges.
    # If an edge is unsimplified, its geometry is a trivial straight line between its incident nodes u and v.
    try:
        geom = data["geometry"]
        linestrings_dict[(u, v)] = geom
    except:
        u_geom = nodes_dict.get(u)
        v_geom = nodes_dict.get(v)
        geom_edge = LineString([u_geom, v_geom])
        linestrings_dict[(u, v)] = geom_edge

output_file = fcd_folder_path + fcd_file + ouput_file_possix + "_" + str(datum) + ".csv"

# New table with GPS positions
# edge = [0 - timestamp,1 - node_from,2 - node_to,3 - segment_length,
# 4- vehicle_id, 5- start_offset_m, 6 - speed_mps, 7 - status, 8 - active]
with open(output_file, 'w', newline='') as csvfile:
    print('Working on interpolating nodes ...')
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['seconds', 'st_y', 'st_x', 'direction', 'speed', 'vehicle_id'])
    for edge in edges:
        timestamp = edge[0]
        date, time = timestamp.split(" ")
        date = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%y%m%d")
        hours, minutes, seconds = time.split(":")
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

        direction = 0
        offset = float(edge[5])
        size = float(edge[3])
        linestring_geometry = linestrings_dict.get((int(edge[1]), int(edge[2])))
        if linestring_geometry and date == datum:
            interpolated_point = linestring_geometry.interpolate(offset / size, normalized=True)
            csv_writer.writerow([total_seconds,
                                 interpolated_point.y,
                                 interpolated_point.x,
                                 direction,
                                 round(float(edge[6]) * 3.6, 2),
                                 edge[4]])

print('Finished in :' + str(datetime.datetime.now() - ct))
print(f"Output is stored in:  {output_file}.")
