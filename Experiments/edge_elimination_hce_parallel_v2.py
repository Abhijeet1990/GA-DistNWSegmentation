"""
Created on Sun July 09:24:00 2021
This code will parallelise the code for evaluating multiple solutions in parallel
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
from system_description_new import SystemDescription
import time
import threading
import collections


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
        threads_f1 = {}
        res_f1 = {}
        for d in range(x.shape[0]):
            threads_f1[d] = threading.Thread(target=self.compute_metric, args=(x[d],res_f1,d))
            threads_f1[d].start()
        for d in range(x.shape[0]):
            threads_f1[d].join()
        #od_f1 = collections.OrderedDict(sorted(res_f1.items()))
        for i in sorted(res_f1.keys()):
            f1.append(res_f1[i])
        out["F"] = anp.column_stack([np.array(f1)])

        # we will add the objective also to evaluate the maximization of load served within each island
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
        #od_g1 = collections.OrderedDict(sorted(res_g1.items()))
        for i in sorted(res_g1.keys()):
            g_temp[i] = res_g1[i]
        g.append(g_temp)

        g_temp = np.zeros(x.shape[0])
        #od_g2 = collections.OrderedDict(sorted(res_g2.items()))
        for i in sorted(res_g1.keys()):
            g_temp[i] = res_g2[i]
        g.append(g_temp)

        g_temp = np.zeros(x.shape[0])
        #od_g3 = collections.OrderedDict(sorted(res_g3.items()))
        for i in sorted(res_g1.keys()):
            g_temp[i] = res_g3[i]
        g.append(g_temp)

        out["G"] = anp.column_stack(np.array(g))
        print(out)


    def compute_metric(self, sol,res,ix):
        edge_index = np.where(sol == 1)
        type_node = nx.get_node_attributes(self.G, 'nodetype')
        weight_label = nx.get_edge_attributes(self.G, 'pflow')
        net_cost = 0
        for i, edge in enumerate(list(self.G.edges)):
            if i in list(edge_index[0]):
                if type_node[edge[0]] == 'Node' and type_node[edge[1]] == 'Node' and weight_label[edge] != 0.0:
                    val = self.G.get_edge_data(*edge)
                    net_cost += val['weights']
        res[ix] = net_cost

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


nsga2_alg = NSGA2(
    pop_size=100,
    n_offsprings=20,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_k_point", prob=0.9, n_points=20),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)

max_no_subgraphs=623
min_no_subgraphs=2
min_node_partition = 2
sd = SystemDescription(_case='hce')
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
                     nsga2_alg,
                     termination=('n_gen', 5000),
                     seed=1,
                     save_history=False)


end = time.time()
time_to_solve = end - start
print('Parallel Solves in '+str(time_to_solve)+' secs')
print('******Graph partition solution NSGA-2*************')
print('Best soln found %s '% res_gp_mo.X)
print('Func Value %s '% res_gp_mo.F)
print('Constraint Value %s '% res_gp_mo.G)
pos = nx.get_node_attributes(G, 'pos')


results={}
if res_gp_mo.X is not None:
    results[str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)] = res_gp_mo.X
    results['score'] = res_gp_mo.F
    scipy.io.savemat('new_problem_hce_result_20cp_100ps_20os_'+str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'.mat',results)


n_evals = np.array([e.evaluator.n_eval for e in res_gp_mo.history])
opt = np.array([e.opt[0].F for e in res_gp_mo.history])

plt.title("Convergence")
plt.plot(n_evals, opt, "--")
plt.yscale("log")
plt.savefig('new_problem_hce_result_20cp_100ps_20os_'+str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'.png',dpi=75)
plt.show()

