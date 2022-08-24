import random

import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
import autograd.numpy as anp
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
import networkx as nx
import re
import sys
from system_description import SystemDescription
import matplotlib.pyplot as plt
from pymoo.configuration import Configuration
Configuration.show_compile_hint = False

# design a graph partition algorithm with the number of partition between a range
# a solution will be indicating whether the line will be open or not

class GraphPartition(Problem):

    def __init__(self, _G, adj_matrix, xcoord, ycoord, partition_size_upper = 20, partition_size_lower=0):
        self.nodes = len(adj_matrix)
        self.adj = adj_matrix
        self.x_loc = xcoord
        self.y_loc = ycoord
        self.np_size_upper = partition_size_upper
        self.np_size_lower = partition_size_lower
        self.G = _G
        super().__init__(n_var=len(list(self.G.edges)), n_obj=1, n_constr=2, xl = 0, xu = 1, type_var= int)
        #super().__init__(xl=0, xu=1,type_var=int)


    def _evaluate(self, x, out, *args, **kwargs):

        # first cost function is to minimize the net loss in edge
        f1 = []
        for d in range(x.shape[0]):
            f1.append(np.sum(x[d,:]))
        out["F"] = anp.column_stack([np.array(f1)])

        # we will add the objective also to evaluate the maximization of load served within each island


        # add the constraint
        g = []
        # the number of sub-graph based on the solution should be within upper and lower limit
        g_temp = np.zeros(x.shape[0])
        for m in range(x.shape[0]):
            edge_index = np.where(x[m] == 1)
            temp_graph = self.G.copy()
            # get the edge index from original graph
            edge_list = list(self.G.edges)
            for i,edge in enumerate(edge_list):
                if i in list(edge_index[0]):
                    temp_graph.remove_edge(edge[0],edge[1])
            subgraph = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]
            g_temp[m] = len(subgraph) - self.np_size_upper
        g.append(g_temp)

        g_temp = np.zeros(x.shape[0])
        for m in range(x.shape[0]):
            edge_index = np.where(x[m] == 1)
            temp_graph = self.G.copy()
            # get the edge index from original graph
            edge_list = list(self.G.edges)
            for i, edge in enumerate(edge_list):
                if i in list(edge_index[0]):
                    temp_graph.remove_edge(edge[0], edge[1])
            subgraph = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]
            g_temp[m] = -len(subgraph) + self.np_size_lower
        g.append(g_temp)


        out["G"] = anp.column_stack(np.array(g))
        print(out)

nsga2_alg = NSGA2(
    pop_size=500,
    n_offsprings=20,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_k_point", prob=0.9, n_points=5),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)

A = np.array([[0,1,1,1,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[1,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
G = nx.from_numpy_array(A)
xcoords=[2,1,3,4,6,5]
ycoords=[5,1,2,4,3,1]

from networkx.generators.random_graphs import erdos_renyi_graph
n = 10
p = 0.2
G = erdos_renyi_graph(n, p)
A = nx.adj_matrix(G).todense()
xcoords=[]
ycoords=[]
for i in range(n):
    xcoords.append(random.randint(1,10))
    ycoords.append(random.randint(1,10))
pos={}
for i in G.nodes:
    pos[i] = (xcoords[i],ycoords[i])
nx.draw_networkx(G,pos)
plt.show()

gp_problem = GraphPartition(G, adj_matrix=A, partition_size_upper=4,partition_size_lower=2,xcoord=np.array(xcoords), ycoord=np.array(ycoords))
res_gp_mo = minimize(gp_problem,
                     nsga2_alg,
                     termination=('n_gen', 5000),
                     seed=1,
                     save_history=True)

print('******Graph partition solution NSGA-2*************')
print('Best soln found %s '% res_gp_mo.X)
print('Func Value %s '% res_gp_mo.F)
print('Constraint Value %s '% res_gp_mo.G)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
if res_gp_mo.X is not None:
    # pic the first soln if there are multiple soln
    if res_gp_mo.X.shape[0] > 1:
        sol = res_gp_mo.X[0]
    else:
        sol = res_gp_mo.X
    edge_remove_index = np.where(np.array(sol) == True)
    edge_list = list(G.edges)
    pos = {}
    for i in G.nodes:
        pos[i] = (xcoords[i],ycoords[i])
    for i, edge in enumerate(edge_list):
        if i in list(edge_remove_index[0]):
            G.remove_edge(edge[0], edge[1])
    colorlist = ['r','g','b','c','m']
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for k,subgraph in enumerate(S):
        for node in subgraph.nodes:
            nx.draw_networkx_nodes(G, pos, [node], node_size=16, node_color=colorlist[k])
        #nx.draw_networkx(subgraph,pos, edge_color=colorlist[k], node_color=colorlist[k])
    colors = ['k' for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, node_size=16, edge_color=colors)
    nx.draw_networkx_labels(G, pos, font_size=10)
plt.show()
print('******Graph partition solution NSGA-2*************')
print('Best soln found %s '% res_gp_mo.X)
print('Func Value %s '% res_gp_mo.F)
print('Func Value %s '% res_gp_mo.X)

import numpy as np
import matplotlib.pyplot as plt

n_evals = np.array([e.evaluator.n_eval for e in res_gp_mo.history])
opt = np.array([e.opt[0].F for e in res_gp_mo.history])

plt.title("Convergence")
plt.plot(n_evals, opt, "--")
plt.yscale("log")
plt.show()