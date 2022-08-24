import win32com.client
import dss_function
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import re

dss = win32com.client.Dispatch('OpenDSSEngine.DSS')

dssText = dss.Text
dssCircuit = dss.ActiveCircuit
dssSolution = dssCircuit.Solution

dss_data_dir = "..\\123Bus_Simple\\"
dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\123Bus_Simple\IEEE123Master.dss'"
dssSolution.Solve()
dssText.Command = r"Buscoords 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\123Bus_Simple\BusCoords.dat'"
lines = dssCircuit.Lines.AllNames
src = []
dest = []
src_x = []
src_y = []
dst_x = []
dst_y = []
n = dssCircuit.NumCktElements
name = np.array("                      ").repeat(n)
busname = np.array("                      ").repeat(n)
busnameto = np.array("                      ").repeat(n)
x = np.zeros(n)
y = np.zeros(n)
xto = np.zeros(n)
yto = np.zeros(n)
distance = np.zeros(n)
nphases = np.zeros(n)
kvbase = np.zeros(n)
I = np.zeros((n,3), dtype=complex)
V = np.zeros((n,3), dtype=complex)
Vto = np.zeros((n,3), dtype=complex)
pflows = np.zeros((n,3), dtype=float)
qflows = np.zeros((n,3), dtype=float)
i = 0

for j in range(0,n):
    el = dssCircuit.CktElements(j)
    if not re.search('^Line', el.Name):
        continue
    name[i] = el.Name
    bus2 = dssCircuit.Buses(re.sub(r"\..*", "", el.BusNames[-1]))
    busnameto[i] = bus2.Name
    xto[i] = bus2.x
    yto[i] = bus2.y
    if bus2.x == 0 or bus2.y == 0: continue  # skip lines without proper bus coordinates
    distance[i] = bus2.distance
    v = np.array(bus2.Voltages)
    nodes = np.array(bus2.nodes)
    kvbase[i] = bus2.kVBase
    nphases[i] = nodes.size
    if nodes.size > 3: nodes = nodes[0:3]
    cidx = 2 * np.array(range(0, min(int(v.size / 2), 3)))
    bus1 = dssCircuit.Buses(re.sub(r"\..*", "", el.BusNames[0]))
    if bus1.x == 0 or bus1.y == 0:
        continue  # skip lines without proper bus coordinates
    busname[i] = bus1.Name
    src.append(busname[i])
    dest.append(busnameto[i])
    src_x.append(bus1.x)
    src_y.append(bus1.y)
    dst_x.append(bus2.x)
    dst_y.append(bus2.y)
    Vto[i, nodes - 1] = v[cidx] + 1j * v[cidx + 1]
    x[i] = bus1.x
    y[i] = bus1.y
    v = np.array(bus1.Voltages)
    V[i, nodes - 1] = v[cidx] + 1j * v[cidx + 1]
    current = np.array(el.Currents)
    I[i, nodes - 1] = current[cidx] + 1j * current[cidx + 1]
    pflows[i, nodes - 1] = (V[i, nodes - 1] * I[i, nodes - 1].conj()).real / 1000
    qflows[i, nodes - 1] = (V[i, nodes - 1] * I[i, nodes - 1].conj()).imag / 1000
    i = i + 1

nodes = []
xcoords = []
ycoords = []
for i,s in enumerate(src):
    if s not in nodes:
        nodes.append(s)
        xcoords.append(src_x[i])
        ycoords.append(src_y[i])
for j,d in enumerate(dest):
    if d not in nodes:
        nodes.append(d)
        xcoords.append(dst_x[j])
        ycoords.append(dst_y[j])

G = nx.Graph()
for n in nodes:
    G.add_node(n)

pflow_list =[]
for ix,pf in enumerate(pflows):
    pflow_list.append(abs(np.average(pflows[ix])))

for s,d,pf in zip(src,dest,pflow_list):
    G.add_edge(s,d, pflow =pf)

edge_attribs = G.edges(data=True)

Adj = np.zeros((len(src),len(src)))

for i,s in enumerate(src):
    for j, d in enumerate(dest):
        match = [item[2]['pflow'] for item in edge_attribs if (str(item[0]) == s and str(item[1]) == d)]
        if len(match) >0:
            Adj[i][j] = match[0]


pos = {}
N = dssCircuit.NumBuses
for i in range(N):
    bus = dssCircuit.Buses(i)
    pos[bus.Name] = (bus.x, bus.y)

fig, axs = plt.subplots(1, 1, figsize=(10, 6))
ax = axs
plt.rcParams.update({'font.size': 20})
nx.draw_networkx_nodes(G, pos, ax=ax,  node_size=16)
nx.draw_networkx_edges(G, pos, ax=ax,  node_size=16)
ax.set_title("Location of the nodes in the distribution grid")
ax.grid()
ax.set_ylabel("Y coordinates")
ax.set_xlabel("X coordinates")
plt.show()


# V1 = dssCircuit.AllNodeVmagPUByPhase(1)
# Dist1 = dssCircuit.AllNodeDistancesByPhase(1)
#
# V2 = dssCircuit.AllNodeVmagPUByPhase(2)
# Dist2 = dssCircuit.AllNodeDistancesByPhase(2)
#
# V3 = dssCircuit.AllNodeVmagPUByPhase(3)
# Dist3 = dssCircuit.AllNodeDistancesByPhase(3)





