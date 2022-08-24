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
import pandas as pd

# assume that we obtained the pgen and qgen dispatch solution for each island, in the MPC they solved, here how do we obtain??
def generate_dss(file_name, edges_removed, Nisland, vsource_bus, supply_bus_ID, pgen, qgen, v_sqr):

    bus_number = 123
    temp_file = open(file_name,'a')
    temp_file.write('\n')

    # disable the lines that were removed due to the partition
    for cc,edge in enumerate(edges_removed):
        temp_file.write('\nDisable Line.L' + str(edge))

    supply_pgen_a = []
    supply_pgen_b = []
    supply_pgen_c = []
    supply_qgen_a = []
    supply_qgen_b = []
    supply_qgen_c = []
    supply_v_a = []
    supply_v_b = []
    supply_v_c = []
    for aa in supply_bus_ID:
        supply_pgen_a.append(pgen[aa - 1][0])
        supply_pgen_b.append(pgen[aa - 1][1])
        supply_pgen_c.append(pgen[aa - 1][2])
        supply_qgen_a.append(qgen[aa - 1][0])
        supply_qgen_b.append(qgen[aa - 1][1])
        supply_qgen_c.append(qgen[aa - 1][2])
        supply_v_a.append(np.sqrt(v_sqr[aa - 1][0]))
        supply_v_b.append(np.sqrt(v_sqr[aa - 1][1]))
        supply_v_c.append(np.sqrt(v_sqr[aa - 1][2]))


    temp_file.write('\n')
    for ii in range(Nisland):
        temp_vec = vsource_bus[ii]
        temp_len = len(temp_vec)
        for kk in range(temp_len):
            temp_bus_id = temp_vec[kk]
            temp_supply_id = supply_bus_ID.index(temp_bus_id)

            # add the first one as the vsource and the rest as the generator
            if kk == 0:
                temp_file.write("\nNew Vsource.sub" + str(temp_bus_id) + "a phases=1 bus1=" + str(
                    temp_bus_id) + ".1 basekv=2.4 angle=0 pu=" + str(supply_v_a[temp_supply_id]))
                temp_file.write("\nNew Vsource.sub" + str(temp_bus_id) + "b phases=1 bus1=" + str(
                    temp_bus_id) + ".2 basekv=2.4 angle=-120 pu=" + str(supply_v_b[temp_supply_id]))
                temp_file.write("\nNew Vsource.sub" + str(temp_bus_id) + "c phases=1 bus1=" + str(
                    temp_bus_id) + ".3 basekv=2.4 angle=120 pu=" + str(supply_v_c[temp_supply_id]))
            else:
                temp_file.write("\nNew Generator.der" + str(temp_bus_id) + "a phases=1 bus1=" + str(
                    temp_bus_id) + ".1 conn=wye kv=2.4 kw=" + str(
                    supply_pgen_a[temp_supply_id] + 0.001) + " kvar=" + str(
                    supply_qgen_a[temp_supply_id] + 0.001) + " model=1")
                temp_file.write("\nNew Generator.der" + str(temp_bus_id) + "b phases=1 bus1=" + str(
                    temp_bus_id) + ".2 conn=wye kv=2.4 kw=" + str(
                    supply_pgen_b[temp_supply_id] + 0.001) + " kvar=" + str(
                    supply_qgen_b[temp_supply_id] + 0.001) + " model=1")
                temp_file.write("\nNew Generator.der" + str(temp_bus_id) + "c phases=1 bus1=" + str(
                    temp_bus_id) + ".3 conn=wye kv=2.4 kw=" + str(
                    supply_pgen_c[temp_supply_id] + 0.001) + " kvar=" + str(
                    supply_qgen_c[temp_supply_id] + 0.001) + " model=1")

    temp_file.write("\n")

    load_fixed_p = pd.read_excel('C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\123Bus_DER\system_data.xlsx', sheet_name='fixed active load')
    load_fixed_q = pd.read_excel('C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\123Bus_DER\system_data.xlsx', sheet_name='fixed reactive load')
    load_info_p = {}
    load_info_q ={}
    for i, j in load_fixed_p.iterrows():
        load_info_p[str(i+1)] = [load_fixed_p['One'][i], load_fixed_p['Two'][i],load_fixed_p['Three'][i]]

    for i, j in load_fixed_p.iterrows():
        load_info_p[str(i+1)] = [load_fixed_q['One'][i], load_fixed_q['Two'][i],load_fixed_q['Three'][i]]

    for k in load_info_p.keys() & load_info_p.keys():
        if load_info_p[k][0] != 0:
            temp_file.write("\nNew Load.S" + str(k) + "a  bus1=" + str(
                k) + ".1 phases=1 conn=wye model=1 kv=2.4 kw=" + str(load_info_p[k][0]) + " kvar=" + str(
                load_info_q[k][0]))
        if load_info_p[k][1] != 0:
            temp_file.write("\nNew Load.S" + str(k) + "b  bus1=" + str(
                k) + ".2 phases=1 conn=wye model=1 kv=2.4 kw=" + str(load_info_p[k][1]) + " kvar=" + str(
                load_info_q[k][1]))
        if load_info_p[k][2] != 0:
            temp_file.write("\nNew Load.S" + str(k) + "c  bus1=" + str(
                k) + ".3 phases=1 conn=wye model=1 kv=2.4 kw=" + str(load_info_p[k]) + " kvar=" + str(
                load_info_q[k][1]))

    temp_file.close()


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

        # Dt 21 july we will add the objective also to evaluate the maximization of load served within each island
        for m in range(x.shape[0]):
            edge_index = np.where(x[m] == 1)
            temp_graph = self.G.copy()
            # get the edge index from original graph
            edge_list = list(self.G.edges)
            for i,edge in enumerate(edge_list):
                if i in list(edge_index[0]):
                    temp_graph.remove_edge(edge[0],edge[1])
            subgraph = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]

            # for each partition solve the power flow by adding a v-source and compute the net load served within each partition

            # assuming these are the list of nodes where there are DERs
            supply_bus_ID = [116, 21, 64, 105, 35, 48, 78, 95]
            bus_island_bin =[]
            line_island_bin = []
            for i,sg in enumerate(subgraph):
                bus_island_bin.append(list(nx.nodes(sg)))
                line_island_bin.apend(list(nx.edges(sg)))

            vsource_bus = []
            for ii in range(len(subgraph)):
                temp_vec = bus_island_bin[ii]
                temp_vsource = []
                for jj in supply_bus_ID:
                    if jj in temp_vec:
                        temp_vsource.append(jj)
                if len(temp_vsource) == 0:
                    temp_vsource.append(temp_vec[0])
                vsource_bus.append(temp_vsource)

                # function to generate the new DSS configuration to run


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

nsga2_alg = NSGA2(
    pop_size=100,
    n_offsprings=20,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_k_point", prob=0.9, n_points=20),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)

max_no_subgraphs=15
min_no_subgraphs=4
min_node_partition = 6
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
                     nsga2_alg,
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
    scipy.io.savemat('new_problem_ieee123bus_der_result_20cp_100ps_'+str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'.mat',results)


n_evals = np.array([e.evaluator.n_eval for e in res_gp_mo.history])
opt = np.array([e.opt[0].F for e in res_gp_mo.history])

plt.title("Convergence")
plt.plot(n_evals, opt, "--")
plt.yscale("log")
plt.savefig('results.png',dpi=75)
plt.show()

