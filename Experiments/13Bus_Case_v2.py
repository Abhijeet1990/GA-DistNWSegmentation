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
from system_description import SystemDescription


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

    def __init__(self, _G, adj_matrix, xcoord, ycoord, partitions = 4, partition_size_upper = 8, partition_size_lower=2):
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

        #third cost is to minimize the spread of a zone in any particular direction in order to make compact zone
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
        #out["F"] = anp.column_stack([np.array(f1), np.array(f3)])
        #out["F"] = anp.column_stack([np.array(f2), np.array(f3)])

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
    paths_between_pairs =[]
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
                    # if finds path between the pair then set true else false
                    for found in path_founds:
                        if found == True:
                            paths_between_pairs.append(True)
                            break
                        else:
                            if path_founds.index(found) == len(path_founds) - 1:
                                paths_between_pairs.append(False)
                                break

    # if any pair is non-contiguous violates
    for test in paths_between_pairs:
        if test == False:
            return M+1

    return M-1

# invert each element of the adj matrix
def invert_elements_adj(adj):
    return np.reciprocal(adj)

nsga2_alg = NSGA2(
    pop_size=100,
    n_offsprings=20,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_k_point", prob=0.9, n_points=5),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)


sd = SystemDescription(_case='ieee13')
G,Adj,xcoords,ycoords = sd.SolveAndCreateGraph()

zones = 5
max_nodes_per_zone = 8
results = {}
gp_problem = GraphPartition(G, adj_matrix=Adj, partitions=zones,partition_size_upper=max_nodes_per_zone,partition_size_lower=1,xcoord=np.array(xcoords), ycoord=np.array(ycoords))
res_gp_mo = minimize(gp_problem,
                     nsga2_alg,
                     termination=('n_gen', 5000),
                     seed=1,
                     save_history=True)
count=1
if res_gp_mo.X is not None:
    for item in res_gp_mo.X:
        print('Solution '+str(count+1)+'\n')
        for i in range(zones):
            items_in_partition = []
            for j in range(len(G.nodes)):
                if item[j*zones +i] == 1:
                    items_in_partition.append(list(G.nodes)[j])
            print('Partition'+str(i+1)+' : '+str(items_in_partition)+'\n')
        count+=1
if res_gp_mo.X is not None:
    results[str(zones)+'_'+str(max_nodes_per_zone)] = res_gp_mo.X
    results['score'] = res_gp_mo.F
    scipy.io.savemat('all_ieee13bus_result_fix.mat',results)

#scipy.io.savemat('nsga_ieee13bus_result.mat',results)
# plot = Scatter()
# plot.add(gp_problem.pareto_front(), plot_type="line", color="black", alpha=0.7)bin
# plot.add(res_gp_mo.F, color="red")
# plot.show()

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


import matplotlib.pyplot as plt
from pymoo.performance_indicator.hv import Hypervolume

# MODIFY - this is problem dependend
ref_point = np.array([1.0, 1.0])

# create the performance indicator object with reference point
metric = Hypervolume(ref_point=ref_point, normalize=False)

# calculate for each generation the HV metric
hv = [metric.calc(f) for f in F]

# visualze the convergence curve
plt.plot(n_evals, hv, '-o', markersize=4, linewidth=2)
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.show()

print('******Graph partition solution NSGA-2*************')
print('Best soln found %s '% res_gp_mo.X)
print('Func Value %s '% res_gp_mo.F)





