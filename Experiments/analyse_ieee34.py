import scipy.io
from system_description import SystemDescription

sd = SystemDescription(_case='ieee123_der')
G,Adj,xcoords,ycoords = sd.SolveAndCreateGraph()

for i in range(Adj.shape[0]):
    for j in range(Adj.shape[1]):
        if Adj[i][j]!=0 and Adj[i][j] != sd.M and i < j:
            print(str(list(G.nodes)[i])+'->'+str(list(G.nodes)[j])+' : '+str(Adj[i][j]))
print(Adj)