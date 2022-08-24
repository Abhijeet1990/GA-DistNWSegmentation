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
import graph_util as gutil

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

class GraphPartition(Problem):

    def __init__(self, adj_matrix, xcoord, ycoord, node_size = 5, partitions = 3, partition_size_upper = 3, partition_size_lower=1):
        self.nodes = node_size
        self.zones = partitions
        self.adj = adj_matrix
        self.x_loc = xcoord
        self.y_loc = ycoord
        self.np_size_upper = partition_size_upper
        self.np_size_lower = partition_size_lower
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
                                if self.adj[j][k] != 0 and self.adj[j][k] < 1000:
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

                # computes the sum of shortest path between nodes within a partition
                #g_temp[m] = shortest_path(np.reciprocal(self.adj), sol, element_ix, j,self.zones) - 1000
                g_temp[m] = shortest_path(self.adj, sol, element_ix, j, self.zones) - 1000

            g.append(g_temp)

        out["G"] = anp.column_stack(np.array(g))
        print(out)

# This should make sure that every pair within the partition are connected
def shortest_path(adj_matrix, sol, element_ix, zone_ix, partitions):
    net_path = 0

    gr = gutil.Graph(len(element_ix))
    gr.graph = adj_matrix

    for i in range(len(element_ix)):
        if sol[element_ix[i]] == 1:
            src_node = int((element_ix[i]-zone_ix)/partitions)
            dist_from_src = gr.dijkstra(src_node)
            for j in range(len(element_ix)):
                if i!=j and sol[element_ix[j]] == 1:
                    dst_node = int((element_ix[j]-zone_ix)/partitions)
                    net_path += dist_from_src[dst_node]

    return net_path


nsga2_alg = NSGA2(
    pop_size=20,
    n_offsprings=10,
    sampling=get_sampling("int_random"),
    crossover=get_crossover("int_sbx", prob=0.9, eta=15),
    mutation=get_mutation("int_pm",eta=20),
    eliminate_duplicates=True)

y = [[0,1,3,4,1000],[1,0,1000,1000,1000],[3,1000,0,2,1000],[4,1000,2,0,2],[1000,1000,1000,2,0]]
gp_problem = GraphPartition(adj_matrix=y,xcoord=[1,2,3,4,5], ycoord=[1,2,3,4,5])
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

# _X = np.row_stack([a.pop.get('X') for a in res_gp_mo.history])
# feasible = np.row_stack([a.pop.get('feasible') for a in res_gp_mo.history])[:,0]
# plotting.plot(_X[feasible], _X[np.logical_not(feasible)], res_gp_mo.X[None,:], labels= ['Feasible','Not feasible','Best'])



# gamethod = get_algorithm('ga',
#                        pop_size = 20,
#                        crossover = get_crossover('int_sbx',prob=1.0,eta=3.0),
#                        mutation = get_mutation('int_pm'),
#                        sampling = get_sampling('int_random'),
#                        eliminate_duplicates=True)
#
# res = minimize(DiscreteProblem(),
#                gamethod,
#                termination=('n_gen',50),
#                seed=1,
#                save_history=True)
#
dpmo = DiscreteProblemMO()
res_mo = minimize(dpmo,
               nsga2_alg,
               termination=('n_gen',50),
               seed=1,
               save_history=True)
#
# print('******Single obj with GA*************')
# print('Best soln found %s '% res.X)
# print('Func Value %s '% res.F)
# print('Constraint violation %s '% res.CV)
# _X = np.row_stack([a.pop.get('X') for a in res.history])
# feasible = np.row_stack([a.pop.get('feasible') for a in res.history])[:,0]
# plotting.plot(_X[feasible], _X[np.logical_not(feasible)], res.X[None,:], labels= ['Feasible','Not feasible','Best'])
#
# print('******Multiple obj with NSGA-2*************')
# print('Best soln found %s '% res_mo.X)
# print('Func Value %s '% res_mo.F)
# print('Constraint violation %s '% res_mo.CV)
#
# _X = np.row_stack([a.pop.get('X') for a in res_mo.history])
# feasible = np.row_stack([a.pop.get('feasible') for a in res_mo.history])[:,0]
# plotting.plot(_X[feasible], _X[np.logical_not(feasible)], res_mo.X[None,:], labels= ['Feasible','Not feasible','Best'])

# plot = Scatter()
# plot.add(dpmo.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res_mo.F, color="red")
# plot.show()



