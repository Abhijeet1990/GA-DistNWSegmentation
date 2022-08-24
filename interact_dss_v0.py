import opendssdirect as dss
import dss_function
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# construct the graph based on the IEEE 123 case and look for partition

###--- read opendss data ---###
dss_data_dir = "..\\123Bus_Simple\\"
dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'
dss.run_command(dss_master_file_dir)

dss_load_coordinates = 'Buscoords ' + dss_data_dir + 'BusCoords.dat ! load in bus coordinates'
dss.run_command(dss_load_coordinates)

circuit = dss.Circuit

#bus_coords = circuit.AllBusDistances()

Bus_name_vec = circuit.AllBusNames()  # read bus name
bus_number = len(Bus_name_vec)
bus_vbase = []
temp_bus_phases = []

# the following code derive bus phases, this might need modification when applied to other systems
for i in range(bus_number):
    circuit.SetActiveBus(Bus_name_vec[i])
    bus_vbase.append(dss.Bus.kVBase())  # get bus base kV
    temp_bus_phases.append(dss.Bus.Nodes())


line_info, switch_info = dss_function.get_lines(dss)
tran_info = dss_function.get_transformer(dss, dss.Circuit)
element_info, element_name, element_capacity, element_number, element_index = dss_function.get_element(dss,
                                                                                                       dss.Circuit,
                                                                                                       None)  # seems not used for now

load_info = []
load_info, total_load_cap = dss_function.get_loads(dss, dss.Circuit)

# Creation of graph
phase=3
df = dss.utils.lines_to_dataframe()
G = nx.Graph()
data = df[['Bus1', 'Bus2']].to_dict(orient="index")

#voltages = dss.Circuit.AllBusVolts()

for name in data:
    line = data[name]
    if ".%s" % phase in line["Bus1"] and ".%s" % phase in line["Bus2"]:
        G.add_edge(line["Bus1"].split(".")[0], line["Bus2"].split(".")[0])
pos = {}
for name in dss.Circuit.AllBusNames():
    dss.Circuit.SetActiveBus("%s" % name)
    if phase in dss.Bus.Nodes():
        index = dss.Bus.Nodes().index(phase)
        re, im = dss.Bus.PuVoltage()[2 * index:2 * index + 2]
        V = abs(complex(re, im))
        D = dss.Bus.Distance()
        pos[dss.Bus.Name()] = (D, V)


# pos = {}
# for name in dss.Circuit.AllBusNames():
#     dss.Circuit.SetActiveBus("%s" % name)
#     names = dss.Circuit.
#     node_list = dss.Bus.Nodes()

fig, axs = plt.subplots(1, 1, figsize=(10, 6))
ax = axs
plt.rcParams.update({'font.size': 20})
nx.draw_networkx_nodes(G, pos, ax=ax,  node_size=16)
nx.draw_networkx_edges(G, pos, ax=ax,  node_size=16)
ax.set_title("Voltage profile plot for phase C")
ax.grid()
ax.set_ylabel("Voltage in p.u.")
ax.set_xlabel("Distances in km")
plt.show()

print('Diagram')