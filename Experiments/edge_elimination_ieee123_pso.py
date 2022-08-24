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
import scipy.io
from system_description import SystemDescription
import time
from pymoo.algorithms.so_pso import PSO, PSOAnimation

# design a graph partition algorithm with the number of partition between a range
# a solution will be indicating whether the line will be open or not

class GraphPartition(Problem):

    def __init__(self, _G, adj_matrix, xcoord, ycoord, partition_size_upper = 20, partition_size_lower=0, min_nodes_partition = 3):
        self.nodes = len(adj_matrix)
        self.adj = adj_matrix
        self.x_loc = xcoord
        self.y_loc = ycoord
        self.np_size_upper = partition_size_upper
        self.np_size_lower = partition_size_lower
        self.min_node_per_partition = min_nodes_partition
        self.G = _G
        super().__init__(n_var=len(list(self.G.edges)), n_obj=1, n_constr=3, xl = 0, xu = 1, type_var= int)
        #super().__init__(xl=0, xu=1,type_var=int)


    def _evaluate(self, x, out, *args, **kwargs):

        # first cost function is to minimize the net loss in edge
        f1 = []
        for d in range(x.shape[0]):
            #f1.append(np.sum(x[d,:]))
            edge_index = np.where(x[d] == 1)
            net_cost = 0
            for i,edge in enumerate(list(self.G.edges)):
                if i in list(edge_index[0]):
                    val = self.G.get_edge_data(*edge)
                    net_cost += val['weights']
            f1.append(net_cost)
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

        # subgraph sizes must be greater than atleast 2 or 3
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
            const = -1
            for sg in enumerate(subgraph):
                if len(list(sg[1].nodes)) < self.min_node_per_partition:
                    const = 1
                    break
            g_temp[m] = const
        g.append(g_temp)



        out["G"] = anp.column_stack(np.array(g))
        print(out)

# nsga2_alg = NSGA2(
#     pop_size=250,
#     n_offsprings=20,
#     sampling=get_sampling("bin_random"),
#     crossover=get_crossover("bin_k_point", prob=0.9, n_points=10),
#     mutation=get_mutation("bin_bitflip"),
#     eliminate_duplicates=True)
pso_alg = PSO(pop_size=500)
max_no_subgraphs=15
min_no_subgraphs=4
min_node_partition = 5
sd = SystemDescription(_case='ieee123_der')
G,Adj,xcoords,ycoords = sd.SolveAndCreateGraph()
sg = [G.subgraph(c) for c in nx.connected_components(G)]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
gp_problem = GraphPartition(G, adj_matrix=Adj,
                            min_nodes_partition=min_node_partition,
                            partition_size_upper=max_no_subgraphs,
                            partition_size_lower=min_no_subgraphs,
                            xcoord=np.array(xcoords),
                            ycoord=np.array(ycoords))

start = time.time()

res_gp_mo = minimize(gp_problem,
                     pso_alg,
                     termination=('n_gen', 5000),
                     seed=1,
                     save_history=True)


end = time.time()
time_to_solve = end - start
print('Solves in '+str(time_to_solve)+' secs')
print('******Graph partition solution NSGA-2*************')
print('Best soln found %s '% res_gp_mo.X)
print('Func Value %s '% res_gp_mo.F)
print('Constraint Value %s '% res_gp_mo.G)
pos = nx.get_node_attributes(G, 'pos')


results={}
if res_gp_mo.X is not None:
    results[str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)] = res_gp_mo.X
    results['score'] = res_gp_mo.F
    scipy.io.savemat('new_problem_ieee123bus_der_result_10cp_250ps_'+str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'.mat',results)

# if res_gp_mo.X is not None:
#     # pic the first soln if there are multiple soln
#     # if res_gp_mo.X.shape[1] > 1:
#     #     sol = res_gp_mo.X[0]
#     # else:
#     sol = res_gp_mo.X
#     edge_remove_index = np.where(np.array(sol) == True)
#     edge_list = list(G.edges)
#     for i, edge in enumerate(edge_list):
#         if i in list(edge_remove_index[0]):
#             G.remove_edge(edge[0], edge[1])
#             print('removes line '+str(edge[0])+'->'+str(edge[1]))
#     colorlist = ['r','g','b','c','m']
#     S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
#     for k,subgraph in enumerate(S):
#         for node in subgraph.nodes:
#             nx.draw_networkx_nodes(G, pos, [node], node_size=16, node_color=colorlist[k])
#         #nx.draw_networkx(subgraph,pos, edge_color=colorlist[k], node_color=colorlist[k])
#     colors = ['k' for u, v in G.edges()]
#     nx.draw_networkx_edges(G, pos, ax=ax, node_size=16, edge_color=colors)
#     nx.draw_networkx_labels(G, pos, font_size=10)
# plt.show()

