"""
Created on Mon June 28 08:01:00 2021
@author: Abhijeet Sahu
"""

import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
import autograd.numpy as anp
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
from pymoo.util import plotting


import sys

class Graph:

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]


    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):

        # Initilaize minimum distance for next node
        min = sys.maxsize
        min_index = None
        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v

        return min_index

    # Funtion that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):

        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)

            # Put the minimum distance vertex in the
            # shotest path tree
            if u is None:
                return dist
            else:
                sptSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and \
                        dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]

        return dist


class DiscreteProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=1, xl = 0, xu = 10, type_var= int)

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = -np.min(x*[3,1], axis = 1 )
        out['G'] = x[:,0] + x[:,1] - 10

class DiscreteProblemMO(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2, xl = 0, xu = 10, type_var= int)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = -np.min(x*[3,1], axis = 1)
        f2 = np.max(x*[2,9], axis =1)

        g1 = x[:,0] + x[:,1] - 10
        g2 = x[:,0] - x[:,1] - 2

        out['F'] = anp.column_stack([f1, f2])
        out['G'] = anp.column_stack([g1, g2])
        print(out)


# design a graph partition algorithm with the number of partition being fixed
# For example there are M nodes in a graph and N partition need to be done
# The decision variable is represented as : one sample for M=5 and N=3, where 1st and 2nd node belong to partition 1
# 3rd and 4th node belong to partition 2 and 5th node belong to partition 3: [ 1 0 0 | 1 0 0 | 0 1 0 | 0 1 0 | 0 0 1]
M = 1000
class GraphPartition(Problem):

    def __init__(self, _G, adj_matrix, xcoord, ycoord, partitions = 3, partition_size_upper = 3, partition_size_lower=1):
        self.nodes = len(adj_matrix)
        self.zones = partitions
        self.adj = adj_matrix
        self.x_loc = xcoord
        self.y_loc = ycoord
        self.np_size_upper = partition_size_upper
        self.np_size_lower = partition_size_lower
        self.G = _G
        super().__init__(n_var=self.nodes*self.zones, n_obj=2, n_constr=self.nodes + self.zones, xl = 0, xu = 1, type_var= int)

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
            src_node = int((element_ix[i] - zone_ix) / partitions)
            for j in range(len(element_ix)):
                if i!=j and sol[element_ix[j]] == 1:
                    dst_node = int((element_ix[j]-zone_ix)/partitions)

                    # get the shortest path between any two nodes
                    paths = nx.all_simple_paths(G, src_node+1, dst_node+1)
                    path_founds = []
                    path_list=[]
                    count=0
                    for path in paths:
                        path_list.append(path)
                        path_founds.append(True)
                        count+=1
                        for item in path:
                            if item != src_node+1 and item != dst_node+1:
                                if sol[element_ix[item - 1]] == 0:
                                    path_founds[count-1] = False
                                    break

                    for found in path_founds:
                        if found == True:
                            return M-1

    return M+1


nsga2_alg = NSGA2(
    pop_size=20,
    n_offsprings=10,
    sampling=get_sampling("int_random"),
    crossover=get_crossover("int_sbx", prob=0.9, eta=15),
    mutation=get_mutation("int_pm",eta=20),
    eliminate_duplicates=True)

import networkx as nx

G = nx.Graph()

y = [[0,3,M,M,M,M],[3,0,13,9,12,M],[M,13,0,M,M,1],[M,9,M,0,7,M],[M,12,M,7,0,4],[M,M,1,M,4,0]]

for i in range(len(y)):
    G.add_node(i+1)

for i in range(len(y)):
    for j in range(len(y)):
        if (y[i][j] != 0 and y[i][j] != M and i < j):
            G.add_edge(i+1,j+1, weights=y[i][j])


edge_attribs = G.edges(data=True)

Adj = np.zeros((len(y),len(y)))

for i in range(len(y)):
    for j in range(len(y)):
        if i == j:
            Adj[list(G.nodes).index(i + 1)][list(G.nodes).index(j + 1)] = 0
        else:
            match = [item[2]['pflow'] for item in edge_attribs if ((item[0] == i+1 and item[1] == j+1) or (item[0] == j+1 and item[1] == i+1))]
            if len(match) >0:
                Adj[list(G.nodes).index(i+1)][list(G.nodes).index(j+1)] = match[0]
            else:
                Adj[list(G.nodes).index(i + 1)][list(G.nodes).index(j + 1)] = M

gp_problem = GraphPartition(G,adj_matrix=Adj,xcoord=[100,200,400,100,300,500], ycoord=[500,300,400,200,100,300])
res_gp_mo = minimize(gp_problem,
               nsga2_alg,
               termination=('n_gen',50),
               seed=1,
               save_history=True)

plot = Scatter()
plot.add(gp_problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res_gp_mo.F, color="red")
plot.show()

print('******Graph partition solution NSGA-2*************')
print('Best soln found %s '% res_gp_mo.X)
print('Func Value %s '% res_gp_mo.F)