import scipy.io
from system_description_new import SystemDescription
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sd = SystemDescription(_case='hce')
G,Adj,xcoords,ycoords = sd.SolveAndCreateGraph()

# res = scipy.io.loadmat(r'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\Snowmass_Cozy\Snowmass_Cozy\new_problem_hce_result_20cp_100ps_20os_623_7_3.mat')
# res_score= res['score']
# res_soln = res['623_7_3']

res = scipy.io.loadmat(r'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\Snowmass_Cozy\Snowmass_Cozy\three_obj_hce_result_20cp_200ps_20os_623_7_15.mat')
res_score= res['score']
res_soln = res['623_7_15']

fig, ax = plt.subplots(1, 1, figsize=(10, 9))
pos = nx.get_node_attributes(G, 'pos')

#nx.draw_networkx(G, pos,with_labels=False,node_size=16)
#plt.show()


# plot the figures
if res_soln is not None:
    sol = res_soln[0]
    type_node = nx.get_node_attributes(G, 'nodetype')
    weight_label = nx.get_edge_attributes(G, 'pflow')
    edge_remove_index = np.where(np.array(sol) == 1)
    edge_list = list(G.edges)
    for i, edge in enumerate(edge_list):
        if i in list(edge_remove_index[0]):
            if type_node[edge[0]] == 'Node' and type_node[edge[1]] == 'Node' and weight_label[edge] != 0.0:
                G.remove_edge(edge[0], edge[1])
                print('removes line '+str(edge[0])+'->'+str(edge[1]))
    colorlist = ['r','g','b','c','m','y','k','aquamarine','mediumseagreen','#00b4d9','#f1b219']

    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    cmap = cm.get_cmap('viridis', len(S) + 1)
    text = 'Total number of microgrids '+str(len(S))
    for k,subgraph in enumerate(S):
        for node in subgraph.nodes:
            nx.draw_networkx_nodes(G, pos, [node], node_size=24, node_color=colorlist[k])
        #nx.draw_networkx(subgraph,pos, edge_color=colorlist[k], node_color=colorlist[k])
    colors = ['k' for u, v in G.edges()]
    ax.set_title(text, fontdict={'fontsize': 10})
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=colors)
    #nx.draw_networkx_labels(G, pos, font_size=10)

plt.savefig('three_obj_hce_grid_7_15.png',dpi=600)
print(res_score)
print(res_soln)