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

    def __init__(self, _G, adj_matrix, xcoord, ycoord, partition_size_upper = 20, partition_size_lower=0, min_nodes_partition = 3,obj_consider=[1,2,3]):
        self.nodes = len(adj_matrix)
        self.adj = adj_matrix
        self.x_loc = xcoord
        self.y_loc = ycoord
        self.np_size_upper = partition_size_upper
        self.np_size_lower = partition_size_lower
        self.min_node_per_partition = min_nodes_partition
        self.G = _G
        self.obj_under_consideration = obj_consider
        super().__init__(n_var=len(list(self.G.edges)), n_obj=3, n_constr=3, xl = 0, xu = 1, type_var= int)
        #super().__init__(xl=0, xu=1,type_var=int)


    def _evaluate(self, x, out, *args, **kwargs):

        # first cost function is to minimize the net loss in edge
        if 1 in self.obj_under_consideration:
            f1 = []
            threads_f1 = {}
            res_f1 = {}
            for d in range(x.shape[0]):
                threads_f1[d] = threading.Thread(target=self.compute_metric, args=(x[d],res_f1,d))
                threads_f1[d].start()
            for d in range(x.shape[0]):
                threads_f1[d].join()
            for i in sorted(res_f1.keys()):
                f1.append(res_f1[i])

        # second cost is to minimize the spread of a zone in any particular direction in order to make compact zone
        if 2 in self.obj_under_consideration or 3 in self.obj_under_consideration:
            f2 = []
            threads_f2 = {}
            res_f2 = {}
            f3=[]
            res_f3={}
            for d in range(x.shape[0]):
                threads_f2[d] = threading.Thread(target=self.compute_metric_loc, args=(x[d],res_f2,res_f3, d))
                threads_f2[d].start()
            for d in range(x.shape[0]):
                threads_f2[d].join()
            for i in sorted(res_f2.keys()):
                f2.append(res_f2[i])
                f3.append(res_f3[i])
        if len(self.obj_under_consideration)==3:
            out["F"] = anp.column_stack([np.array(f1),np.array(f2),np.array(f3)])
        elif len(self.obj_under_consideration)==2:
            if 1 in self.obj_under_consideration and 2 in self.obj_under_consideration:
                out["F"] = anp.column_stack([np.array(f1), np.array(f2)])
            elif  1 in self.obj_under_consideration and 3 in self.obj_under_consideration:
                out["F"] = anp.column_stack([np.array(f1), np.array(f3)])
            elif  2 in self.obj_under_consideration and 3 in self.obj_under_consideration:
                out["F"] = anp.column_stack([np.array(f2), np.array(f3)])
        else:
            if 1 in self.obj_under_consideration:
                out["F"] = anp.column_stack([np.array(f1)])
            elif 2 in self.obj_under_consideration:
                out["F"] = anp.column_stack([np.array(f2)])
            elif 3 in self.obj_under_consideration:
                out["F"] = anp.column_stack([np.array(f3)])

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

    def compute_metric_loc(self,sol,res,res2,ix):
        edge_index = np.where(sol == 1)
        type_node = nx.get_node_attributes(self.G, 'nodetype')
        weight_label = nx.get_edge_attributes(self.G, 'pflow')
        net_cost = 0
        net_cost_2=0
        pos = nx.get_node_attributes(G, 'pos')
        temp_graph = self.G.copy()
        # get the edge index from original graph
        edge_list = list(self.G.edges)
        for i, edge in enumerate(edge_list):
            if i in list(edge_index[0]):
                if type_node[edge[0]] == 'Node' and type_node[edge[1]] == 'Node' and weight_label[edge] != 0.0:
                    temp_graph.remove_edge(edge[0], edge[1])
        subgraph = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]
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

        for i, si in enumerate(subgraph):
            for j, sj in enumerate(subgraph):
                if j > i:
                    net_cost_2 += abs(len(list(si.nodes))-len(list(sj.nodes)))
        res2[ix] = net_cost_2

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

ps = 200
os = 20
cp = 20
term_limit=1000
nsga2_alg = NSGA2(
    pop_size=ps,
    n_offsprings=os,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_k_point", prob=0.9, n_points=cp),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)

obj_consideration=[3]
max_no_subgraphs=623
min_no_subgraphs=7
min_node_partition = 13
sd = SystemDescription(_case='hce')
G,Adj,xcoords,ycoords = sd.SolveAndCreateGraph()
sg = [G.subgraph(c) for c in nx.connected_components(G)]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
gp_problem = GraphPartition(G, adj_matrix=Adj,
                            min_nodes_partition=min_node_partition,
                            partition_size_upper=max_no_subgraphs,
                            partition_size_lower=min_no_subgraphs,
                            xcoord=np.array(xcoords),
                            ycoord=np.array(ycoords),
                            obj_consider=obj_consideration)

start = time.time()

res_gp_mo = minimize(gp_problem,
                     nsga2_alg,
                     termination=('n_gen', term_limit),
                     seed=1,
                     save_history=True)


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
    results['time'] = time_to_solve
    results['termination'] = term_limit
    scipy.io.savemat('f3_'+str(cp)+'_'+str(ps)+'_'+str(os)+'_'+str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'.mat',results)


n_evals = np.array([e.evaluator.n_eval for e in res_gp_mo.history])
opt = np.array([e.opt[0].F for e in res_gp_mo.history])

plt.title("Convergence")
plt.plot(n_evals, opt, "--")
plt.yscale("log")
plt.savefig('f3_'+str(cp)+'_'+str(ps)+'_'+str(os)+'_'+str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'_v1.png',dpi=75)
#plt.show()

n_evals = []    # corresponding number of function evaluations\
F = []          # the objective space values in each generation
cv = []         # constraint violation in each generation


# iterate over the deepcopies of algorithms
for algorithm in res_gp_mo.history:

    # store the number of function evaluations
    n_evals.append(algorithm.evaluator.n_eval)

    # retrieve the optimum from the algorithm
    opt = algorithm.opt

    # store the least contraint violation in this generation
    cv.append(opt.get("CV").min())

    # filter out only the feasible and append
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append(_F)


from pymoo.performance_indicator.hv import Hypervolume

ref_point = np.array([0.0, 0.0, 0.0])

# create the performance indicator object with reference point
metric = Hypervolume(ref_point=ref_point, normalize=False)

# calculate for each generation the HV metric
hv=[]
for f in F:
    if f.shape[0] >0:
        hv.append(metric.calc(f))
    else:
        hv.append(0)

#hv = [metric.calc(f) for f in F]

# visualze the convergence curve
plt.plot(n_evals, hv, '-o', markersize=4, linewidth=2)
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.savefig('f3_'+str(cp)+'_'+str(ps)+'_'+str(os)+'_'+str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'_v2.png',dpi=75)


