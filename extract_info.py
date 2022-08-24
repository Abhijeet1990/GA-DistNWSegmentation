"""
Created on Mon July 10 10:45:00 2021
@author: Abhijeet Sahu
This code we formulate the graph based on the voltage profile, power flows as the adjacency matrix and use NSGA-2 with
the proposed graph partition algorithm
"""

import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
import autograd.numpy as anp
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
from pymoo.util import plotting
import opendssdirect as dss
import win32com.client
import networkx as nx
import re
import sys
import scipy.io
M = 1000000
## Interaction with Open DSS to obtain the graph, node and edge features

dss = win32com.client.Dispatch('OpenDSSEngine.DSS')

dssText = dss.Text
dssCircuit = dss.ActiveCircuit
dssSolution = dssCircuit.Solution

dss_data_dir = "..\\123Bus_Simple\\"
dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\Shared_Codes\13Bus\IEEE13Nodeckt.dss'"
dssSolution.Solve()
# dssText.Command = r"Buscoords 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\123Bus_Simple\BusCoords.dat'"
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
    #if bus2.x == 0 or bus2.y == 0: continue  # skip lines without proper bus coordinates
    distance[i] = bus2.distance
    v = np.array(bus2.Voltages)
    nodes = np.array(bus2.nodes)
    kvbase[i] = bus2.kVBase
    nphases[i] = nodes.size
    if nodes.size > 3: nodes = nodes[0:3]
    cidx = 2 * np.array(range(0, min(int(v.size / 2), 3)))
    bus1 = dssCircuit.Buses(re.sub(r"\..*", "", el.BusNames[0]))
    # if bus1.x == 0 or bus1.y == 0:
    #     continue  # skip lines without proper bus coordinates
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
    G.add_edge(s,d, weights = pf,pflow =pf)

edge_attribs = G.edges(data=True)

Adj = np.zeros((len(nodes),len(nodes)))


for s in list(G.nodes):
    for d in list(G.nodes):
        if s == d:
            Adj[list(G.nodes).index(s)][list(G.nodes).index(d)] = 0
        else:
            match = [item[2]['weights'] for item in edge_attribs if ((str(item[0]) == s and str(item[1]) == d) or (str(item[0]) == d and str(item[1]) == s))]
            if len(match) >0:
                Adj[list(G.nodes).index(s)][list(G.nodes).index(d)] = match[0]
            else:
                Adj[list(G.nodes).index(s)][list(G.nodes).index(d)] = M

# load the results
res = scipy.io.loadmat('nsga_ieee13bus_result_5_5.mat')
res_5_5 = res['5_5']
partition = 5
count=0
for item in res_5_5:
    print ('solution '+str(count+1)+'\n')
    for i in range(partition):
        items_in_partition=[]
        for j in range(len(G.nodes)):
            if item[j*partition + i] == 1:
                items_in_partition.append(list(G.nodes)[j])
        print('Partition '+str(i+1)+' : '+str(items_in_partition) +'\n')
    count+=1

res = scipy.io.loadmat('nsga_ieee13bus_result_6_4.mat')
res_6_4 = res['6_4']
partition = 6
count=0
for item in res_6_4:
    print ('solution '+str(count+1)+'\n')
    for i in range(partition):
        items_in_partition=[]
        for j in range(len(G.nodes)):
            if item[j*partition + i] == 1:
                items_in_partition.append(list(G.nodes)[j])
        print('Partition '+str(i+1)+' : '+str(items_in_partition) +'\n')
    count+=1


