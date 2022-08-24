import opendssdirect as dss
import dss_function
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# construct the graph based on the IEEE 123 case and look for partition

###--- read opendss data ---###
dss_data_dir = "..\\123Bus\\"
dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master_0_withload.dss'
dss.run_command(dss_master_file_dir)
circuit = dss.Circuit
Bus_name_vec = circuit.AllBusNames()  # read bus name
bus_number = len(Bus_name_vec)
bus_vbase = []
temp_bus_phases = []

# the following code derive bus phases, this might need modification when applied to other systems
for i in range(bus_number):
    circuit.SetActiveBus(Bus_name_vec[i])
    bus_vbase.append(dss.Bus.kVBase())  # get bus base kV
    temp_bus_phases.append(dss.Bus.Nodes())
bus_phases = []
for i in range(bus_number):
    temp_index = Bus_name_vec.index(str(i + 1))
    tempvec = [0, 0, 0]
    for j in range(len(temp_bus_phases[temp_index])):
        tempvec[temp_bus_phases[temp_index][j] - 1] = 1
    bus_phases.append(tempvec)

line_info, switch_info = dss_function.get_lines(dss)
tran_info = dss_function.get_transformer(dss, dss.Circuit)
element_info, element_name, element_capacity, element_number, element_index = dss_function.get_element(dss,
                                                                                                       dss.Circuit,
                                                                                                       None)  # seems not used for now

load_info = []
load_info, total_load_cap = dss_function.get_loads(dss, dss.Circuit)
pv_info = dss_function.get_Generator(dss)

set_bus_ESS = []
set_bus_DG_diesel = []
set_bus_DG_pv = []
set_bus_DG_wind = []
set_bus_DG = []
set_bus_HMES = []
hems_node_info = []
# hems_data_file = pd.read_csv(dss_data_dir + 'control_node_list.csv', header=None)  # required step
# hems_data_file = pd.read_csv(dss_data_dir + 'system_data.xlsx', header='HEMS ')  # required step
bus_id_seq = pd.read_excel(dss_data_dir + 'system_data.xlsx', sheet_name='bus_id_to_name')
hems_bus = pd.read_excel(dss_data_dir + 'system_data.xlsx', sheet_name='HEMS indicator')
ESS_cap = pd.read_excel(dss_data_dir + 'system_data.xlsx', sheet_name='Utility ESS')
# DG info for IEEE 123 bus system is stored in external excel. For utility feeders, DG info may be stored in pv_info
if len(pv_info) < 1:
    DG_cap = pd.read_excel(dss_data_dir + 'system_data.xlsx', sheet_name='Max DG capacity')
load_fixed_p = pd.read_excel(dss_data_dir + 'system_data.xlsx', sheet_name='fixed active load')
load_fixed_q = pd.read_excel(dss_data_dir + 'system_data.xlsx', sheet_name='fixed reactive load')
cap_home_pv = pd.read_excel(dss_data_dir + 'random_data.xlsx', sheet_name='PV_cap')
cap_home_ess = pd.read_excel(dss_data_dir + 'random_data.xlsx', sheet_name='Home_ESS')

cap_ESS = []
cap_DG_diesel = []
cap_DG_pv = []
cap_DG_wind = []
cap_DG = []

for i in range(bus_number - 1):  # construct bus sets for ESS, DG
    if ESS_cap.iloc[i, 0] != 0:
        set_bus_ESS.append(i)
        cap_ESS.append((ESS_cap.iloc[i, :] / 3).tolist())
    if DG_cap.iloc[i, 0] != 0:
        set_bus_DG.append(i)
        cap_DG.append((DG_cap.iloc[i, :] / 3).tolist())
        if DG_cap.iloc[i, 0] == 720:
            set_bus_DG_pv.append(i)
            cap_DG_pv.append((DG_cap.iloc[i, :] / 3).tolist())
        elif DG_cap.iloc[i, 0] == 900:
            set_bus_DG_wind.append(i)
            cap_DG_wind.append((DG_cap.iloc[i, :] / 3).tolist())
        elif DG_cap.iloc[i, 0] == 1200:
            set_bus_DG_diesel.append(i)
            cap_DG_diesel.append((DG_cap.iloc[i, :] / 3).tolist())
        else:
            print('Undefined generator')
cap_DG = np.array(cap_DG)
cap_ESS = np.array(cap_ESS)

set_bus_supply = set_bus_DG[:]
set_bus_supply.append(115)
set_bus_vsource = set_bus_supply
ess_ut_number = len(set_bus_ESS)
dg_number = len(set_bus_DG) * 3

set_home_pv = []
set_home_ess = []
cap_home_ess_P_low = np.zeros(len(cap_home_pv))
cap_home_ess_P_up = np.zeros(len(cap_home_pv))
cap_home_ess_S = np.zeros(len(cap_home_pv))
cap_home_ess_soc = np.zeros(len(cap_home_pv))

for count in range(len(cap_home_pv)):
    # hems_node_info.append(hems_data_file.iloc[count][0])
    if cap_home_pv.iloc[count, 0] > 0:
        set_home_pv.append(count)
    if cap_home_ess.iloc[count, 0] > 0:
        set_home_ess.append(count)

cap_home_ess_P_low[set_home_ess] = -2.5
cap_home_ess_P_up[set_home_ess] = 2.5
cap_home_ess_S[set_home_ess] = 5
cap_home_ess_soc[set_home_ess] = 5

home_pv_number = len(set_home_pv)
home_ess_number = len(set_home_ess)
cap_home_pv = np.array(cap_home_pv)
cap_home_ess = np.array(cap_home_ess)

# Creation of graph
df = dss.utils.lines_to_dataframe()
G = nx.Graph()
data = df[['Bus1', 'Bus2']].to_dict(orient="index")
phase = 3

#
# buses = dss.Circuit.AllBusNames()
# dss.Circuit.SetActiveBus("%s" % buses[0])
# nodes=dss.Bus.Nodes()
# index = dss.Bus.Nodes().index(phase)
# re, im = dss.Bus.PuVoltage()[2 * index:2 * index + 2]
# V = abs(complex(re, im))
# D = dss.Bus.Distance()
#
# dss.Circuit.SetActiveBus("%s" % buses[1])
# nodes=dss.Bus.Nodes()
# index = dss.Bus.Nodes().index(phase)
# re, im = dss.Bus.PuVoltage()[2 * index:2 * index + 2]
# V = abs(complex(re, im))
# D = dss.Bus.Distance()

voltages = dss.Circuit.AllBusVolts()


for name in data:
    line = data[name]
    if ".%s" % phase in line["Bus1"] and ".%s" % phase in line["Bus2"]:
        G.add_edge(line["Bus1"].split(".")[0], line["Bus2"].split(".")[0])
    # pos = {}
    # for name in dss.Circuit.AllBusNames():
    #     dss.Circuit.SetActiveBus("%s" % name)
    #     if phase in dss.Bus.Nodes():
    #         index = dss.Bus.Nodes().index(phase)
    #         re, im = dss.Bus.PuVoltage()[2 * index:2 * index + 2]
    #         V = abs(complex(re, im))
    #         D = dss.Bus.Distance()
    #         pos[dss.Bus.Name()] = (D, V)

pos= nx.spring_layout(G)
fig, axs = plt.subplots(1, 1, figsize=(10, 6))
ax = axs
plt.rcParams.update({'font.size': 20})
nx.draw_networkx_nodes(G, pos, ax=ax,  node_size=16)
nx.draw_networkx_edges(G, pos, ax=ax,  node_size=16)
nx.draw_networkx_labels(G, pos)
# ax.set_title("Voltage profile plot for phase A")
# ax.grid()
# ax.set_ylabel("Voltage in p.u.")
# ax.set_xlabel("Distances in km")
plt.show()

print('Plotting for Phase 1')