"""
Created on Mon Aug 2 09:24:00 2021
This code is the graph partition and visualization for Snowmass Cozy feeder
@author: Abhijeet Sahu
"""

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
from system_description_snowmass_cozy import SystemDescription
import time
import threading
import collections
import matplotlib.cm as cm

# design a graph partition algorithm with the number of partition between a range
# a solution will be indicating whether the line will be open or not

class GraphPartition(Problem):

    def __init__(self, _G, adj_matrix, xcoord, ycoord, _lode_info,partition_size_upper = 20, partition_size_lower=0, min_nodes_partition = 3):
        self.nodes = len(adj_matrix)
        self.adj = adj_matrix
        self.x_loc = xcoord
        self.y_loc = ycoord
        self.lode_info = _lode_info
        self.np_size_upper = partition_size_upper
        self.np_size_lower = partition_size_lower
        self.min_node_per_partition = min_nodes_partition
        self.G = _G
        super().__init__(n_var=len(list(self.G.edges)), n_obj=3, n_constr=3, xl = 0, xu = 1, type_var= int)
        #super().__init__(xl=0, xu=1,type_var=int)


    def _evaluate(self, x, out, *args, **kwargs):

        # first cost function is to minimize the net loss in edge
        f1 = []
        threads_f1 = {}
        res_f1 = {}
        for d in range(x.shape[0]):
            threads_f1[d] = threading.Thread(target=self.compute_metric_first_objective, args=(x[d],res_f1,d))
            threads_f1[d].start()
        for d in range(x.shape[0]):
            threads_f1[d].join()
        for i in sorted(res_f1.keys()):
            f1.append(res_f1[i])

        # second cost is to minimize the spread of a zone in any particular direction in order to make compact zone
        f2 = []
        threads_f2 = {}
        res_f2 = {}

        # third cost is to minimize the net difference in the number of nodes in different zones
        f3=[]
        res_f3={}

        # fourth cost is to minimize the net difference of the amount of real-power load served within each partition
        f4=[]
        res_f4={}

        for d in range(x.shape[0]):
            threads_f2[d] = threading.Thread(target=self.compute_metric_second_third_fourth_objective, args=(x[d],res_f2,res_f3,res_f4, d))
            threads_f2[d].start()
        for d in range(x.shape[0]):
            threads_f2[d].join()
        for i in sorted(res_f2.keys()):
            f2.append(res_f2[i])
            f3.append(res_f3[i])
            f4.append(res_f4[i])

        out["F"] = anp.column_stack([np.array(f1),np.array(f2),np.array(f3),np.array(f4)])

        # add the constraint
        g = []
        # the number of sub-graph based on the solution should be within upper and lower limit
        g_temp = np.zeros(x.shape[0])
        threads_g1={}
        res_g1 = {}
        res_g2 = {}
        res_g3 = {}
        for m in range(x.shape[0]):
            threads_g1[m] = threading.Thread(target=self.compute_constraint, args=(x[m], res_g1, res_g2,res_g3,m))
            threads_g1[m].start()
        for m in range(x.shape[0]):
            threads_g1[m].join()
        for i in sorted(res_g1.keys()):
            g_temp[i] = res_g1[i]
        g.append(g_temp)

        g_temp = np.zeros(x.shape[0])
        for i in sorted(res_g1.keys()):
            g_temp[i] = res_g2[i]
        g.append(g_temp)

        g_temp = np.zeros(x.shape[0])
        for i in sorted(res_g1.keys()):
            g_temp[i] = res_g3[i]
        g.append(g_temp)

        out["G"] = anp.column_stack(np.array(g))
        print(out)


    def compute_metric_first_objective(self, sol,res,ix):
        edge_index = np.where(sol == 1)
        type_node = nx.get_node_attributes(self.G, 'nodetype')
        weight_label = nx.get_edge_attributes(self.G, 'pflow')
        net_cost = 0

        # This line ensures the lines are removed if they are not directly connected to a load or generator
        for i, edge in enumerate(list(self.G.edges)):
            if i in list(edge_index[0]):
                if type_node[edge[0]] == 'Node' and type_node[edge[1]] == 'Node' and weight_label[edge] != 0.0:
                    val = self.G.get_edge_data(*edge)
                    net_cost += val['weights']
        res[ix] = net_cost

    def compute_metric_second_third_fourth_objective(self,sol,res,res2,res3,ix):
        edge_index = np.where(sol == 1)
        type_node = nx.get_node_attributes(self.G, 'nodetype')
        weight_label = nx.get_edge_attributes(self.G, 'pflow')
        net_cost = 0
        pos = nx.get_node_attributes(G, 'pos')
        temp_graph = self.G.copy()
        # get the edge index from original graph
        edge_list = list(self.G.edges)
        for i, edge in enumerate(edge_list):
            if i in list(edge_index[0]):
                # This line ensures the lines are removed if they are not directly connected to a load or generator
                if type_node[edge[0]] == 'Node' and type_node[edge[1]] == 'Node' and weight_label[edge] != 0.0:
                    temp_graph.remove_edge(edge[0], edge[1])
        subgraph = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]

        # the second objective function computation
        for i,sg in enumerate(subgraph):
            x_coord = []
            y_coord = []
            for node in list(sg.nodes):
                x_coord.append(pos[node][0])
                y_coord.append(pos[node][1])
            if (max(x_coord) - min(x_coord)) > (max(y_coord) - min(y_coord)):
                net_cost += max(x_coord) - min(x_coord)
            elif (max(y_coord) - min(y_coord)) > (max(x_coord) - min(x_coord)):
                net_cost += max(y_coord) - min(y_coord)
        res[ix] = net_cost

        # the third objective function computation
        net_cost_2 = 0
        for i, si in enumerate(subgraph):
            for j, sj in enumerate(subgraph):
                if j > i:
                    net_cost_2 += abs(len(list(si.nodes))-len(list(sj.nodes)))
        res2[ix] = net_cost_2

        # the fourth objective
        net_cost_3 = 0
        load_partition_wise ={}
        for i, si in enumerate(subgraph):
            for node in list(si.nodes):
                if node in load_info.keys():
                    if i not in load_partition_wise.keys():
                        load_partition_wise[i] = self.lode_info[node][1]
                    else:
                        load_partition_wise[i] += self.lode_info[node][1]
        for i, (k,v) in enumerate(load_partition_wise.items()):
            for j, (k1,v1) in enumerate(load_partition_wise.items()):
                if j > i:
                    net_cost_3 += abs(v1 - v)
        res3[ix] = net_cost_3



    def compute_constraint(self,sol,res,res2,res3,ix):
        edge_index = np.where(sol == 1)
        type_node = nx.get_node_attributes(self.G, 'nodetype')
        weight_label = nx.get_edge_attributes(self.G, 'pflow')
        temp_graph = self.G.copy()
        # get the edge index from original graph
        edge_list = list(self.G.edges)
        for i, edge in enumerate(edge_list):
            if i in list(edge_index[0]):
                if type_node[edge[0]] == 'Node' and type_node[edge[1]] == 'Node' and weight_label[edge] != 0.0:
                    temp_graph.remove_edge(edge[0], edge[1])
        subgraph = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]
        res[ix] = len(subgraph) - self.np_size_upper
        res2[ix] = -len(subgraph) + self.np_size_lower
        const = -1
        for sg in enumerate(subgraph):
            if len(list(sg[1].nodes)) < self.min_node_per_partition:
                const = 1
                break
        res3[ix] = const

# if one need to save solution history for evaluation of convergence
save_soln_history= True
ps=500
os=20
cp=20

nsga2_alg = NSGA2(
    pop_size=ps,
    n_offsprings=os,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_k_point", prob=0.9, n_points=cp),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)

max_no_subgraphs=623
min_no_subgraphs=7
min_node_partition = 7
power_file=r'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\Notebooks\Snowmass_D1_Sub_D1__D7_11_D8__EXP_ElemPowers.CSV'
sd = SystemDescription(_power_file=power_file,_case='hce')
G,Adj,xcoords,ycoords,load_info,gen_info = sd.SolveAndCreateGraph()
sg = [G.subgraph(c) for c in nx.connected_components(G)]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
gp_problem = GraphPartition(G, adj_matrix=Adj,_lode_info=load_info,
                            min_nodes_partition=min_node_partition,
                            partition_size_upper=max_no_subgraphs,
                            partition_size_lower=min_no_subgraphs,
                            xcoord=np.array(xcoords),
                            ycoord=np.array(ycoords))

start = time.time()

res_gp_mo = minimize(gp_problem,
                     nsga2_alg,
                     termination=('n_gen', 1000),
                     seed=1,
                     save_history=save_soln_history)


end = time.time()
time_to_solve = end - start
print('Parallel Solves in '+str(time_to_solve)+' secs')
print('******Graph partition solution NSGA-2*************')
print('Best soln found %s '% res_gp_mo.X)
print('Func Value %s '% res_gp_mo.F)
print('Constraint Value %s '% res_gp_mo.G)
pos = nx.get_node_attributes(G, 'pos')

# visualize the result
results={}
if res_gp_mo.X is not None:
    sol = res_gp_mo.X[0]
    type_node = nx.get_node_attributes(G, 'nodetype')
    weight_label = nx.get_edge_attributes(G, 'pflow')
    edge_remove_index = np.where(np.array(sol) == 1)
    edge_list = list(G.edges)
    for i, edge in enumerate(edge_list):
        if i in list(edge_remove_index[0]):
            if type_node[edge[0]] == 'Node' and type_node[edge[1]] == 'Node' and weight_label[edge] != 0.0:
                G.remove_edge(edge[0], edge[1])
                print('removes line ' + str(edge[0]) + '->' + str(edge[1]))
    colorlist = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'mediumseagreen', '#00b4d9', '#f1b219']

    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    cmap = cm.get_cmap('viridis', len(S) + 1)
    text = 'Total number of microgrids ' + str(len(S))
    for k, subgraph in enumerate(S):
        for node in subgraph.nodes:
            nx.draw_networkx_nodes(G, pos, [node], node_size=24, node_color=colorlist[k])
        # nx.draw_networkx(subgraph,pos, edge_color=colorlist[k], node_color=colorlist[k])
    colors = ['k' for u, v in G.edges()]
    ax.set_title(text, fontdict={'fontsize': 10})
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=colors)
    plt.savefig('four_obj_hce_grid_'+str(cp)+'cp_'+str(ps)+'ps_'+str(os)+'os_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'.png', dpi=600)
else:
    print('No solution obtained with the given configuration')

# visualize the convergence result
if save_soln_history:
    n_evals = np.array([e.evaluator.n_eval for e in res_gp_mo.history])
    opt = np.array([e.opt[0].F for e in res_gp_mo.history])

    plt.title("Convergence")
    plt.plot(n_evals, opt, "--")
    plt.yscale("log")
    plt.savefig('four_obj_hce_result_'+str(cp)+'cp_'+str(ps)+'ps_'+str(os)+'os_'+str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'.png',dpi=75)
    plt.show()
else:
    print('Saving history was not enabled')

