import scipy.io
from system_description import SystemDescription
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sd = SystemDescription(_case='ieee123')
G,Adj,xcoords,ycoords = sd.SolveAndCreateGraph()

res = scipy.io.loadmat(r'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\123Bus_DER\new_problem_ieee123bus_der_result_5cp_500ps_15_4_5.mat')
res_score= res['score']
res_soln = res['15_4_5']

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pos = nx.get_node_attributes(G, 'pos')
# plot the figures
if res_soln is not None:
    sol = res_soln[0]
    edge_remove_index = np.where(np.array(sol) == 1)
    edge_list = list(G.edges)
    for i, edge in enumerate(edge_list):
        if i in list(edge_remove_index[0]):
            G.remove_edge(edge[0], edge[1])
            print('removes line '+str(edge[0])+'->'+str(edge[1]))
    colorlist = ['r','g','b','c','m','y','k','w','r']

    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    cmap = cm.get_cmap('viridis', len(S) + 1)
    text = 'Total number of microgrids '+str(len(S))
    for k,subgraph in enumerate(S):
        for node in subgraph.nodes:
            nx.draw_networkx_nodes(G, pos, [node], node_size=24, node_color=colorlist[k])
        #nx.draw_networkx(subgraph,pos, edge_color=colorlist[k], node_color=colorlist[k])
    colors = ['k' for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, node_size=24, edge_color=colors)
    nx.draw_networkx_labels(G, pos, font_size=10)
    ax.set_title(text, fontdict={'fontsize': 10})
    # ax.grid()
    ax.set_ylabel('Y coordinates')
    ax.set_xlabel('X coordinates')
plt.show()

print(res_score)
print(res_soln)