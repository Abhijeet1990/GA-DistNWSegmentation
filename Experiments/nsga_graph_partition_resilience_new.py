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

def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    #if not graph.has_key(start):
    #   return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


def find_all_connected_nodes(key, bus_graph_copy, connected_nodes_list=[]):
    connected_nodes_list_prev = connected_nodes_list
    if key not in connected_nodes_list:
        connected_nodes_list = connected_nodes_list + [key]
    for _key in bus_graph_copy[key]:
        if _key not in connected_nodes_list and _key != key:
            connected_nodes_list = connected_nodes_list + [_key]
    # print("connected_nodes_list: ", connected_nodes_list)
    if len(connected_nodes_list) == len(connected_nodes_list_prev):
        return connected_nodes_list
    for _key in bus_graph_copy[key]:
        if _key != key:
            connected_nodes_list = find_all_connected_nodes(_key, bus_graph_copy, connected_nodes_list)

    return connected_nodes_list


def get_path_redundancy(bus_graph_final, gen_nodes, gen_nodes_new, gen_node_cap, diesel_left, coeff_b, time_step,
                        active_nodes, ess_left, node_dict, bat_s_cap, p_cr):
    load_nodes = list(bus_graph_final.keys())
    load_nodes.remove(0)
    load_nodes.sort()

    #print(gen_nodes)
    #print(load_nodes)

    diesel_required = coeff_b * time_step * gen_node_cap

    p_cr = np.maximum(p_cr, 0.01)
    critical_energy_required = coeff_b * time_step * np.sum(p_cr, axis=1)

    path_redundancy_val = 0
    for load_bus in np.arange(len(load_nodes)):
        load_bus_name = load_nodes[load_bus]
        for gen_bus in np.arange(len(gen_nodes_new)):
            gen_bus_name = gen_nodes_new[gen_bus]
            _path = find_shortest_path(bus_graph_final, load_bus_name, gen_bus_name)
            if _path is not None:
                path_redundancy_val += np.minimum(diesel_left[gen_nodes[gen_bus], 0] /
                                                  critical_energy_required[load_bus], 1)\
                                       / np.maximum(len(_path), 1)
        for load_bus_1 in np.arange(len(load_nodes)):
            load_bus_1_name = load_nodes[load_bus_1]
            _path = find_shortest_path(bus_graph_final, load_bus_name, load_bus_1_name)
            if _path is not None:
                if bat_s_cap[load_bus_1, 0] > 0.1:
                    path_redundancy_val += np.minimum(ess_left[load_bus_1] /
                                                      critical_energy_required[load_bus], 1)\
                                           / np.maximum(len(_path), 1)

    path_redundancy_val = path_redundancy_val / (len(load_nodes) * (len(gen_nodes)+len(load_nodes)))
    return path_redundancy_val


def get_min_feasible_subnetworks(bus_graph_final, gen_nodes_new, gen_node_cap, p_cr, diesel_left, coeff_b, time_step, node_dict, ess_left, bat_p_cap):
    load_nodes = list(bus_graph_final.keys())
    load_nodes.remove(0)
    load_nodes.sort()

    gen_node_cap = [gen_node_cap[i] * np.minimum(diesel_left[i, 0] / (gen_node_cap[i] * coeff_b * time_step), 1)
                    for i in np.arange(len(gen_node_cap)) if np.abs(gen_node_cap[i]) > 0.1]
    p_cr = p_cr.tolist()

    # Dictionary of gens and loads
    gen_dict = {}
    for ind in np.arange(len(gen_nodes_new)):
        gen_dict[gen_nodes_new[ind]] = gen_node_cap[ind]

    load_dict = {}
    for ind in np.arange(len(load_nodes)):
        load_dict[load_nodes[ind]] = p_cr[ind]

    ess_p_dict = {}
    for ind in np.arange(len(load_nodes)):
        ess_p_dict[load_nodes[ind]] = np.minimum(ess_left[ind] / time_step, bat_p_cap[ind, 0])

    # Dictionary of sub-networks, with each key having the list of its buses
    subnetwork_dict = {}
    subnetwork_cnt = 0
    subnetwork_dict[subnetwork_cnt] = []
    for key in bus_graph_final.keys():
        # print("key: ", key)
        # print("subnetwork_dict: ", subnetwork_dict)
        if (subnetwork_cnt == 0) and (subnetwork_dict[subnetwork_cnt] == []):
            connected_nodes_list = find_all_connected_nodes(key, bus_graph_final)
            subnetwork_dict[subnetwork_cnt] = connected_nodes_list
        else:
            flag = 0
            for cnt in np.arange(subnetwork_cnt + 1):
                # print("cnt: ", cnt)
                if key in subnetwork_dict[cnt]:
                    flag = 1
                    break
            if flag == 0:
                subnetwork_cnt += 1
                connected_nodes_list = find_all_connected_nodes(key, bus_graph_final)
                subnetwork_dict[subnetwork_cnt] = connected_nodes_list
    #print("\nNo. of subnetworks: {}".format(subnetwork_cnt+1))
    # for key in subnetwork_dict.keys():
    #     print("\nsubnetwork : ", key)
    #     print([node_dict[i] for i in subnetwork_dict[key]])

    # Find number of feasible subnetworks
    num_feas_networks = []
    feas_networks = 0
    # print("\nfkpnkvnoqav")
    for cnt in np.arange(subnetwork_cnt + 1):
        net_gen = 0
        for _bus in subnetwork_dict[cnt]:
            if (_bus in gen_nodes_new) and (_bus not in load_nodes):
                net_gen += gen_dict[_bus] + ess_p_dict[_bus]
                # print(" +: ", net_gen)
            elif (_bus not in gen_nodes_new) and (_bus in load_nodes):
                net_gen += -load_dict[_bus] + ess_p_dict[_bus]
                # print(" -: ", net_gen)
            elif (_bus in gen_nodes_new) and (_bus in load_nodes):
                net_gen += gen_dict[_bus] - load_dict[_bus] + ess_p_dict[_bus]
                # print("+-: ", net_gen)
        #print(net_gen)
        if net_gen >= 0:
            feas_networks += 1
        # print("net_gen: ", net_gen)

    num_feas_networks.append(feas_networks)
    # print("num_feas_networks: ", num_feas_networks)

    return min(num_feas_networks) / len(load_dict.keys())


def get_input_res(bus_graph, active_nodes, node_dict, gen_nodes, gen_node_cap, diesel_left,
                  ess_left, p_cr, coeff_b, time_step, p_pv, bat_s_cap, bat_p_cap):
    active_nodes = [h[:-2] for h in active_nodes]

    active_ind = []
    for ind in np.arange(len(node_dict)):
        if node_dict[ind] in active_nodes or node_dict[ind] == '116':
            active_ind.append(ind)

    gens = [active_nodes[i] for i in gen_nodes]
    gen_nodes_new = []
    for key in node_dict.keys():
        if node_dict[key] in gens:
            gen_nodes_new.append(key)

    # Calculate inputs

    path_redundancy = get_path_redundancy(bus_graph, gen_nodes, gen_nodes_new, gen_node_cap, diesel_left,
                                          coeff_b, time_step, active_nodes, ess_left, node_dict, bat_s_cap,
                                          p_cr + p_pv)

    min_feasible_subnetworks = get_min_feasible_subnetworks(bus_graph, gen_nodes_new, gen_node_cap, p_cr[:, 0] + p_pv[:, 0], diesel_left, coeff_b, time_step, node_dict, ess_left, bat_p_cap)

    #print("\nPATH_REDUNDANCY: ", path_redundancy)
    #print("MIN_FEASIBLE_SUBNETWORKS: ", min_feasible_subnetworks)

    return path_redundancy, min_feasible_subnetworks


def remove_duplicate_nodes(nodes):
    node_list = []
    for node in nodes:
        if node not in node_list:
            node_list.append(node)
    return node_list

# design a graph partition algorithm with the number of partition between a range
# a solution will be indicating whether the line will be open or not

class GraphPartition(Problem):

    def __init__(self, _G, adj_matrix, xcoord, ycoord, _lode_info,_node_dict,partition_size_upper = 20, partition_size_lower=0, min_nodes_partition = 3):
        self.nodes = len(adj_matrix)
        self.adj = adj_matrix
        self.x_loc = xcoord
        self.y_loc = ycoord
        self.lode_info = _lode_info
        self.np_size_upper = partition_size_upper
        self.np_size_lower = partition_size_lower
        self.min_node_per_partition = min_nodes_partition
        self.node_dict = _node_dict
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

        # compute the resilience metrics
        threads_f5 = {}
        f5=[]
        res_r1={}
        f6=[]
        res_r2={}
        for d in range(x.shape[0]):
            threads_f5[d] = threading.Thread(target=self.compute_resilience_metric, args=(x[d],res_r1,res_r2, d))
            threads_f5[d].start()
        for d in range(x.shape[0]):
            threads_f5[d].join()
        for i in sorted(res_r1.keys()):
            f5.append(-res_r1[i]) # since they need to be maximized
            f6.append(-res_r2[i]) # since they need to be maximized

        #out["F"] = anp.column_stack([np.array(f1),np.array(f2),np.array(f3),np.array(f4)])
        #out["F"] = anp.column_stack([np.array(f1), np.array(f2), np.array(f3), np.array(f4), np.array(f5), np.array(f6)])
        out["F"] = anp.column_stack([ np.array(f2), np.array(f3), np.array(f5), np.array(f6)])

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

    def compute_resilience_metric(self, sol, res_1r, res_2r, ix):
        temp_graph = self.G.copy()
        edge_index = np.where(sol == 1)
        type_node = nx.get_node_attributes(self.G, 'nodetype')
        # get the edge index from original graph
        edge_list = list(self.G.edges)
        for i, edge in enumerate(edge_list):
            if i in list(edge_index[0]):
                temp_graph.remove_edge(edge[0], edge[1])
        bus_graph = {}
        for i, (k, v) in enumerate(temp_graph.degree._nodes.items()):
            key_bus_graph = list(self.node_dict.keys())[list(self.node_dict.values()).index(k)]
            list_dest = []
            for j, (k2, v2) in enumerate(v.items()):
                key_bus_to_graph = list(self.node_dict.keys())[list(self.node_dict.values()).index(k2)]
                list_dest.append(key_bus_to_graph)
            bus_graph[key_bus_graph] = list_dest

        ###########
        active_nodes = ['1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '16', '17', '19', '20',
                        '22', '24', '28', '29', '30', '31', '32', '33', '34', '35', '37', '38', '39',
                        '41', '42', '43', '45', '46', '47', '48', '49', '50', '51', '52', '53', '55',
                        '56', '58', '59', '60', '62', '63', '64', '65', '66', '68', '69', '70', '71',
                        '73', '74', '75', '76', '77', '79', '80', '82', '83', '84', '85', '86', '87',
                        '88', '90', '92', '94', '95', '96', '98', '99', '100', '102', '103', '104',
                        '106', '107', '109', '111', '112', '113', '114']  # gens+loads

        ###########
        gens = ['28', '35', '47', '60', '76', '98', '109', '34', '53', '64', '51', '80']  # only gens

        gen_nodes = []
        for __ind in np.arange(len(bus_graph)):
            if node_dict[__ind] in gens:
                gen_nodes.append(__ind)
        gen_nodes = remove_duplicate_nodes(gen_nodes)

        gen_node_cap = np.zeros(len(bus_graph))
        for gen__ind in np.arange(len(bus_graph)):
            if gen__ind in gen_nodes:
                gen_node_cap[gen__ind] = 100

        ###########
        diesel_left = np.zeros((len(bus_graph), 1))
        for gen__ind in np.arange(len(bus_graph)):
            if gen__ind in gen_nodes:
                diesel_left[gen__ind, 0] = 100

        ###########
        ess_left = np.ones(len(bus_graph))  # in kWh
        for i, j in node_dict.items():
            if list(node_dict.keys())[list(node_dict.values()).index(j)] in active_nodes:
                ess_left[i] = 20
            else:
                ess_left[i] = 0
        ###########
        p_cr = np.random.rand(len(bus_graph), 2)
        for i, j in node_dict.items():
            if list(node_dict.keys())[list(node_dict.values()).index(j)] in active_nodes:
                p_cr[i] = 5
            else:
                p_cr[i] = 0

        ###########
        coeff_b = 0.05

        ###########
        time_step = 0.25

        ###########
        p_pv = np.random.rand(len(bus_graph), 2)
        for i, j in node_dict.items():
            if list(node_dict.keys())[list(node_dict.values()).index(j)] in active_nodes:
                p_pv[i] = 50
            else:
                p_pv[i] = 0

        ###########
        bat_s_cap = np.ones((len(bus_graph), 1))
        for i, j in node_dict.items():
            if list(node_dict.keys())[list(node_dict.values()).index(j)] in active_nodes:
                bat_s_cap[i] = 40
            else:
                bat_s_cap[i] = 0

        ###########
        bat_p_cap = np.ones((len(bus_graph), 1))
        for i, j in node_dict.items():
            if list(node_dict.keys())[list(node_dict.values()).index(j)] in active_nodes:
                bat_p_cap[i] = 20
            else:
                bat_p_cap[i] = 0

        ###########
        input_dict = {
            'bus_graph': bus_graph,
            'active_nodes': active_nodes,
            'node_dict': node_dict,
            'gen_nodes': gen_nodes,
            'gen_node_cap': gen_node_cap,
            'p_pv': p_pv,
            'diesel_left': diesel_left,
            'ess_left': ess_left,
            'p_cr': p_cr,
            'coeff_b': coeff_b,
            'time_step': time_step,
            'bat_s_cap': bat_s_cap,
            'bat_p_cap': bat_p_cap,
        }
        path_redundancy, min_feasible_subnetworks = get_input_res(**input_dict)
        res_1r[ix] = path_redundancy
        res_2r[ix] = min_feasible_subnetworks

    def compute_constraint(self,sol,res,res2,res3,ix):
        edge_index = np.where(sol == 1)
        type_node = nx.get_node_attributes(self.G, 'nodetype')
        weight_label = nx.get_edge_attributes(self.G, 'pflow')
        temp_graph = self.G.copy()
        # get the edge index from original graph
        edge_list = list(self.G.edges)
        for i, edge in enumerate(edge_list):
            if i in list(edge_index[0]):
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
ps=50
os=20
cp=20

nsga2_alg = NSGA2(
    pop_size=ps,
    n_offsprings=os,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_k_point", prob=0.9, n_points=cp),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)

max_no_subgraphs = 123
min_no_subgraphs = 4
min_node_partition = 3
power_file = r'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\Notebooks\ieee123_EXP_ElemPowers.CSV'
sd = SystemDescription(_power_file=power_file,_case='ieee123_der')
# compute the node_dict, bus_graph and active_nodes
node_dict =  node_dict = {0: '116', 1: '1', 2: '3', 3: '5', 4: '7', 5: '8', 6: '9', 7: '13', 8: '14', 9: '15', 10: '18',
                 11: '19', 12: '21', 13: '23', 14: '25', 15: '26', 16: '27', 17: '28', 18: '29', 19: '30', 20: '31',
                 21: '34', 22: '35', 23: '36', 24: '38', 25: '40', 26: '42', 27: '44', 28: '45', 29: '47', 30: '49',
                 31: '50', 32: '51', 33: '52', 34: '53', 35: '54', 36: '55', 37: '57', 38: '58', 39: '60', 40: '62',
                 41: '63', 42: '64', 43: '65', 44: '67', 45: '68', 46: '69', 47: '70', 48: '72', 49: '73', 50: '74',
                 51: '76', 52: '77', 53: '78', 54: '80', 55: '81', 56: '82', 57: '84', 58: '86', 59: '87', 60: '89',
                 61: '91', 62: '93', 63: '95', 64: '97', 65: '98', 66: '99', 67: '100', 68: '120', 69: '101', 70: '102',
                 71: '103', 72: '105', 73: '106', 74: '108', 75: '109', 76: '110', 77: '112', 78: '113', 79: '115',
                 80: '118', 81: '119', 82: '56', 83: '83', 84: '117', 85: '2', 86: '4', 87: '6', 88: '10', 89: '11',
                 90: '12', 91: '16', 92: '17', 93: '20', 94: '22', 95: '24', 96: '32', 97: '33', 98: '37', 99: '39',
                 100: '41', 101: '43', 102: '46', 103: '48', 104: '59', 105: '66', 106: '71', 107: '75', 108: '79',
                 109: '85', 110: '88', 111: '90', 112: '92', 113: '94', 114: '96', 115: '104', 116: '107', 117: '111',
                 118: '114', 119: '121', 120: '61', 121: '123', 122: '122'}

G,Adj,xcoords,ycoords,load_info,gen_info = sd.SolveAndCreateGraph()

sg = [G.subgraph(c) for c in nx.connected_components(G)]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
gp_problem = GraphPartition(G, adj_matrix=Adj,_lode_info=load_info,_node_dict=node_dict,
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
    plt.savefig('res_obj_123_grid_'+str(cp)+'cp_'+str(ps)+'ps_'+str(os)+'os_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'.png', dpi=600)
else:
    print('No solution obtained with the given configuration')

# visualize the convergence result
if save_soln_history:
    n_evals = np.array([e.evaluator.n_eval for e in res_gp_mo.history])
    opt = np.array([e.opt[0].F for e in res_gp_mo.history])
    plt.title("Convergence")
    plt.plot(n_evals, opt, "--")
    plt.yscale("log")
    plt.savefig('res_obj_123_result_'+str(cp)+'cp_'+str(ps)+'ps_'+str(os)+'os_'+str(max_no_subgraphs)+'_'+str(min_no_subgraphs)+'_'+str(min_node_partition)+'.png',dpi=75)
    plt.show()
else:
    print('Saving history was not enabled')

