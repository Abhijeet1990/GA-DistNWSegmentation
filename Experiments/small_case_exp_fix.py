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



# design a graph partition algorithm with the number of partition being fixed
# For example there are M nodes in a graph and N partition need to be done
# The decision variable is represented as : one sample for M=5 and N=3, where 1st and 2nd node belong to partition 1
# 3rd and 4th node belong to partition 2 and 5th node belong to partition 3: [ 1 0 0 | 1 0 0 | 0 1 0 | 0 1 0 | 0 0 1]

# design a graph partition algorithm with the number of partition being fixed
# For example there are M nodes in a graph and N partition need to be done
# The decision variable is represented as : one sample for M=5 and N=3, where 1st and 2nd node belong to partition 1
# 3rd and 4th node belong to partition 2 and 5th node belong to partition 3: [ 1 0 0 | 1 0 0 | 0 1 0 | 0 1 0 | 0 0 1]
M = 1000000
class GraphPartition(Problem):

    def __init__(self, _G, adj_matrix, xcoord, ycoord, partitions = 5, partition_size_upper = 20, partition_size_lower=0):
        self.nodes = len(adj_matrix)
        self.zones = partitions
        self.adj = adj_matrix
        self.x_loc = xcoord
        self.y_loc = ycoord
        self.np_size_upper = partition_size_upper
        self.np_size_lower = partition_size_lower
        self.G = _G
        super().__init__(n_var=self.nodes*self.zones, n_obj=3, n_constr=self.nodes + 2*self.zones, xl = 0, xu = 1, type_var= int)
        #super().__init__(xl=0, xu=1,type_var=int)
    def _evaluate(self, x, out, *args, **kwargs):

        # first cost function is to minimize the net loss in edge values
        f1 = []
        for d in range(x.shape[0]):
            net_first_cost = 0
            for i in range(self.zones):
                T = [x[d,a] for a in range(i, x.shape[1], self.zones)]
                for j in range(len(T)):
                    if T[j] == 1:
                        for k in range(len(T)):
                            if j != k and T[k] == 0:
                                if self.adj[j][k] != 0 and self.adj[j][k] < M:
                                    net_first_cost += self.adj[j][k]
            f1.append(net_first_cost/2)

        # second cost is to minimize the net difference in the number of nodes in different zones
        f2 = []
        for d in range(x.shape[0]):
            net_second_cost=0
            for i in range(self.zones):
                TS = [x[d,a] for a in range(i, x.shape[1], self.zones)]
                ts_one_count = sum(map(lambda m: m == 1, TS))
                for j in range(i,self.zones):
                    TD = [x[d,a] for a in range(j, x.shape[1], self.zones)]
                    td_one_count = sum(map(lambda n: n == 1, TD))
                    net_second_cost += abs(ts_one_count - td_one_count)
            f2.append(net_second_cost)

        # third cost is to minimize the spread of a zone in any particular direction in order to make compact zone
        f3 = []
        for d in range(x.shape[0]):
            net_third_cost = 0
            for i in range(self.zones):
                T = [x[d, a] for a in range(i, x.shape[1], self.zones)]
                x_p =[]
                y_p = []
                for j in range(len(T)):
                    if T[j] == 1:
                       x_p.append(self.x_loc[j])
                       y_p.append(self.y_loc[j])
                if len(x_p)!=0 and len(y_p)!=0:
                    if (max(x_p) - min(x_p)) > (max(y_p) - min(y_p)):
                        net_third_cost += max(x_p) - min(x_p)
                    elif (max(y_p) - min(y_p)) > (max(x_p) - min(x_p)):
                        net_third_cost += max(y_p) - min(y_p)
            f3.append(net_third_cost)

        out["F"] = anp.column_stack([np.array(f1), np.array(f2), np.array(f3)])
        #out["F"] = anp.column_stack([np.array(f1), np.array(f2)])

        # add the constraint
        g = []

        # This ensures that a node cannot belong to more than one partition/zone
        for i in range(self.nodes):
            g_temp = np.sum(x[:,self.zones*i : self.zones*(i+1)],axis=1) - 1
            g.append(g_temp)

        for i in range(self.nodes):
            g_temp = -np.sum(x[:,self.zones*i : self.zones*(i+1)],axis=1) + 1
            g.append(g_temp)

        # This ensures that a zone can have a maximum of Np_upper
        for j in range(self.zones):
            g_temp = np.sum(x[:,[i for i in range(self.nodes * self.zones) if (i-j)%self.zones == 0]],axis=1) - self.np_size_upper
            g.append(g_temp)

        # This ensures that a zone need to have a minimum of Np_lower
        for j in range(self.zones):
            g_temp = -np.sum(x[:,[i for i in range(self.nodes * self.zones) if (i-j)%self.zones == 0]],axis=1) + self.np_size_lower
            g.append(g_temp)

        # ensuring the nodes in a zone are contiguous, i.e. none of the nodes within zones are separated by a node in a different partition
        # Logic: The sum of shortest path between any two nodes within a partition is less than a large number.
        for j in range(self.zones):
            g_temp = np.zeros(x.shape[0]) # pop_size
            for m in range(x.shape[0]):
                sol = x[m]
                element_ix = [i for i in range(self.nodes * self.zones) if (i-j)%self.zones == 0]
                g_temp[m] = shortest_paths_within_zone(self.G, sol, element_ix, j, self.zones) - M

            g.append(g_temp)

        out["G"] = anp.column_stack(np.array(g))
        print(out)

def shortest_paths_within_zone(G, sol, element_ix,zone_ix,partitions):
    for i in range(len(element_ix)):
        if sol[element_ix[i]] == 1:
            src_node_ix = int((element_ix[i] - zone_ix) / partitions)
            src_node = list(G.nodes)[src_node_ix]
            for j in range(len(element_ix)):
                if i!=j and sol[element_ix[j]] == 1:
                    dst_node_ix = int((element_ix[j]-zone_ix)/partitions)
                    dst_node = list(G.nodes)[dst_node_ix]
                    # get the shortest path between any two nodes
                    paths = nx.all_simple_paths(G, src_node, dst_node)
                    path_founds = []
                    path_list=[]
                    count=0
                    for path in paths:
                        path_list.append(path)
                        path_founds.append(True)
                        count+=1
                        for item in path:
                            if item != src_node and item != dst_node:
                                intermediary_node_ix = list(G.nodes).index(item)
                                if sol[element_ix[intermediary_node_ix]] == 0:
                                    path_founds[count-1] = False
                                    break

                    for found in path_founds:
                        if found == True:
                            return M-1

    return M+1

# invert each element of the adj matrix
def invert_elements_adj(adj):
    return np.reciprocal(adj)

nsga2_alg = NSGA2(
    pop_size=50,
    n_offsprings=20,
    sampling=get_sampling("int_random"),
    crossover=get_crossover("int_sbx", prob=0.9, eta=15),
    mutation=get_mutation("int_pm",eta=20),
    eliminate_duplicates=True)


## Interaction with Open DSS to obtain the graph, node and edge features

dss = win32com.client.Dispatch('OpenDSSEngine.DSS')

dssText = dss.Text
dssCircuit = dss.ActiveCircuit
dssSolution = dssCircuit.Solution

dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\34Bus\ieee34Mod1.dss'"
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



# y = [[0,1,3,4,1000],[1,0,1000,1000,1000],[3,1000,0,2,1000],[4,1000,2,0,2],[1000,1000,1000,2,0]]
gp_problem = GraphPartition(G,adj_matrix=Adj,xcoord=np.array(xcoords), ycoord=np.array(ycoords))
res_gp_mo = minimize(gp_problem,
               nsga2_alg,
               termination=('n_gen',5000),
               seed=1,
               save_history=True)

# plot = Scatter()
# plot.add(gp_problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res_gp_mo.F, color="red")
# plot.show()

print('******Graph partition solution NSGA-2*************')
print('Best soln found %s '% res_gp_mo.X)
print('Func Value %s '% res_gp_mo.F)





