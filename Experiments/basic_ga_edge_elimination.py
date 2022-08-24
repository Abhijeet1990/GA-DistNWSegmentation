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
from pymoo.algorithms.so_genetic_algorithm import GA


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
        super().__init__(n_var=len(list(self.G.edges)), n_obj=1, n_constr=3, xl = 0, xu = 1, type_var= int)
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

        out["F"] = anp.column_stack([np.array(f1)])
        #out["F"] = anp.column_stack([np.array(f3),np.array(f4)])
        #out["F"] = anp.column_stack([np.array(f1), np.array(f2), np.array(f4)])
        #out["F"] = anp.column_stack([np.array(f1),np.array(f2),np.array(f3),np.array(f4)])
        #out["F"] = anp.column_stack([np.array(f1), np.array(f2), np.array(f3)])

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
                #if type_node[edge[0]] == 'Node' and type_node[edge[1]] == 'Node' and weight_label[edge] != 0.0:
                val = self.G.get_edge_data(*edge)
                net_cost += val['weights']
        res[ix] = net_cost

    def compute_metric_second_third_fourth_objective(self,sol,res,res2,res3,ix):
        edge_index = np.where(sol == 1)
        type_node = nx.get_node_attributes(self.G, 'nodetype')
        weight_label = nx.get_edge_attributes(self.G, 'pflow')
        net_cost = 0
        pos = nx.get_node_attributes(self.G, 'pos')
        temp_graph = self.G.copy()
        # get the edge index from original graph
        edge_list = list(self.G.edges)
        for i, edge in enumerate(edge_list):
            if i in list(edge_index[0]):
                # This line ensures the lines are removed if they are not directly connected to a load or generator
                #if type_node[edge[0]] == 'Node' and type_node[edge[1]] == 'Node' and weight_label[edge] != 0.0:
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
                #if type_node[edge[0]] == 'Node' and type_node[edge[1]] == 'Node' and weight_label[edge] != 0.0:
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




# from pymoo.model.crossover import Crossover
# from pymoo.model.mutation import Mutation
# from pymoo.model.sampling import Sampling
#
#
# class MySampling(Sampling):
#
#     def _do(self, problem, n_samples, **kwargs):
#         X = np.full((n_samples, problem.n_var), False, dtype=np.bool)
#
#         for k in range(n_samples):
#             I = np.random.permutation(problem.n_var)[:problem.n_max]
#             X[k, I] = True
#
#         return X
#
#
# class BinaryCrossover(Crossover):
#     def __init__(self):
#         super().__init__(2, 1)
#
#     def _do(self, problem, X, **kwargs):
#         n_parents, n_matings, n_var = X.shape
#
#         _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)
#
#         for k in range(n_matings):
#             p1, p2 = X[0, k], X[1, k]
#
#             both_are_true = np.logical_and(p1, p2)
#             _X[0, k, both_are_true] = True
#
#             n_remaining = problem.n_max - np.sum(both_are_true)
#
#             I = np.where(np.logical_xor(p1, p2))[0]
#
#             S = I[np.random.permutation(len(I))][:n_remaining]
#             _X[0, k, S] = True
#
#         return _X
#
#
# class MyMutation(Mutation):
#     def _do(self, problem, X, **kwargs):
#         for i in range(X.shape[0]):
#             X[i, :] = X[i, :]
#             is_false = np.where(np.logical_not(X[i, :]))[0]
#             is_true = np.where(X[i, :])[0]
#             X[i, np.random.choice(is_false)] = True
#             X[i, np.random.choice(is_true)] = False
#
#         return X
#

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.optimize import minimize

max_no_subgraphs=13
min_no_subgraphs=2
min_node_partition = 3
power_file=r'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\Notebooks\IEEE13Nodeckt_EXP_ElemPowers.CSV'
#power_file=r'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\Notebooks\ieee34-1_EXP_ElemPowers.CSV'
#power_file=r'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\Notebooks\ieee123_EXP_ElemPowers.CSV'
sd = SystemDescription(_power_file=power_file,_case='ieee13_der')
G,Adj,xcoords,ycoords,load_info,gen_info = sd.SolveAndCreateGraph()
sg = [G.subgraph(c) for c in nx.connected_components(G)]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
gp_problem = GraphPartition(G, adj_matrix=Adj,_lode_info=load_info,
                            min_nodes_partition=min_node_partition,
                            partition_size_upper=max_no_subgraphs,
                            partition_size_lower=min_no_subgraphs,
                            xcoord=np.array(xcoords),
                            ycoord=np.array(ycoords))

# algorithm = GA(
#     pop_size=100,
#     sampling=MySampling(),
#     crossover=BinaryCrossover(),
#     mutation=MyMutation(),
#     eliminate_duplicates=True)
cp=20
ps=200
os=40

algorithm = GA(
    pop_size=ps,
    n_offsprings=os,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_k_point", prob=0.9, n_points=cp),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)

start = time.time()

res = minimize(gp_problem,
                     algorithm,
                     termination=('n_gen', 1000),
                     seed=1,
                     save_history=True)

print("Function value: %s" % res.F[0])
print("Subset:", np.where(res.X)[0])


