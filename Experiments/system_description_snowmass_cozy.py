import opendssdirect as dss
import win32com.client
import networkx as nx
import numpy as np
import re
import pandas as pd

class SystemDescription:
    def __init__(self, _power_file, _case='hce', bigM=1000000):
        self.dss = win32com.client.Dispatch('OpenDSSEngine.DSS')
        self.dssText = self.dss.Text
        self.dssCircuit = self.dss.ActiveCircuit
        self.dssSolution = self.dssCircuit.Solution
        self.case = _case
        self.M = bigM
        self.power_file = _power_file

    def SolveAndCreateGraph(self):
        if self.case == 'ieee13_der':
            self.dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\13Bus_DER\IEEE13Nodeckt.dss'"
        elif self.case == 'ieee13':
            self.dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\13Bus\IEEE13Nodeckt.dss'"
        elif self.case == 'ieee34':
            self.dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\34Bus\ieee34Mod1.dss'"
        elif self.case == 'ieee34_der':
            self.dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\34Bus\ieee34Mod_Der.dss'"
        elif self.case == 'ieee123':
            self.dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\123Bus_Simple\IEEE123Master.dss'"
            self.dssText.Command = r"Buscoords 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\123Bus_Simple\BusCoords.dat'"
        elif self.case == 'ieee123_der':
            self.dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\123Bus_DER\ieee123Master_8.dss'"
        elif self.case == 'hce':
            self.dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\Snowmass_Cozy\Snowmass_Cozy\Master_qsts_2020.dss'"
            n = self.dssCircuit.NumCktElements
            for j in range(0, n):
                el = self.dssCircuit.CktElements(j)
                if re.search('^SwtControl', el.Name):
                    self.dssCircuit.SetActiveElement(el.Name)
                    # close all the switch controls
                    self.dssCircuit.SwtControls.Action = 2
                    # print(self.dssCircuit.ActiveElement.IsOpen(fr,to))
        elif self.case == 'hce_mod':
            self.dssText.Command = r"Compile 'C:\Users\asahu\Desktop\Proj2_REORG\Experiments\PYMOO\pymoo\pymoo\usage\Snowmass_Cozy_Modified\Snowmass_Cozy\Master_qsts_2020.dss'"
            # n = self.dssCircuit.NumCktElements
            # for j in range(0, n):
            #     el = self.dssCircuit.CktElements(j)
            #     if re.search('^SwtControl', el.Name):
            #         self.dssCircuit.SetActiveElement(el.Name)
            #         # close all the switch controls
            #         self.dssCircuit.SwtControls.Action = 2
                    # print(self.dssCircuit.ActiveElement.IsOpen(fr,to))

        self.dssSolution.Solve()

        #self.dssText.Command = r"export elempowers"
        #self.dssText.Command = r"export voltages"

        lines = self.dssCircuit.Lines.AllNames
        src = []
        dest = []
        src_x = []
        src_y = []
        dst_x = []
        dst_y = []
        n = self.dssCircuit.NumCktElements
        name = np.array("                      ").repeat(n)
        busname = np.array("                      ").repeat(n)
        busnameto = np.array("                      ").repeat(n)
        x = np.zeros(n)
        y = np.zeros(n)
        xto = np.zeros(n)
        yto = np.zeros(n)
        distance = np.zeros(n)
        nphases = np.zeros(n)
        kvbase = np.zeros(n)

        I = np.zeros((n, 3), dtype=complex)
        V = np.zeros((n, 3), dtype=complex)
        Vto = np.zeros((n, 3), dtype=complex)
        pflows = np.zeros((n, 3), dtype=float)
        qflows = np.zeros((n, 3), dtype=float)
        i = 0

        node_type = {}
        load_info = {}
        gen_info={}

        for j in range(0, n):
            el = self.dssCircuit.CktElements(j)
            if re.search('^SwtControl', el.Name):
                self.dssCircuit.SetActiveElement(el.Name)

            if not re.search('^Line', el.Name) and not re.search('^Transformer', el.Name) and not re.search(
                    '^SwtControl', el.Name) and not re.search('^RegControl', el.Name) and not re.search('^Reactor',
                                                                                                        el.Name):
                print(el.Name)
                if 'Load' in el.Name:
                    node_name = self.dssCircuit.Buses(re.sub(r"\..*", "", el.BusNames[-1])).Name
                    node_type[node_name] = 'Load'
                    load_info[node_name] = el.Name
                elif 'Generator' in el.Name:
                    node_name = self.dssCircuit.Buses(re.sub(r"\..*", "", el.BusNames[-1])).Name
                    node_type[node_name] = 'Generator'
                    gen_info[node_name] = el.Name
                elif 'Vsource' in el.Name:
                    node_name = self.dssCircuit.Buses(re.sub(r"\..*", "", el.BusNames[-1])).Name
                    node_type[node_name] = 'Source'
                continue

            name[i] = el.Name
            bus2 = self.dssCircuit.Buses(re.sub(r"\..*", "", el.BusNames[-1]))
            busnameto[i] = bus2.Name
            xto[i] = bus2.x
            yto[i] = bus2.y
            # if bus2.x == 0 or bus2.y == 0: continue  # skip lines without proper bus coordinates
            distance[i] = bus2.distance
            v = np.array(bus2.Voltages)
            nodes = np.array(bus2.nodes)
            kvbase[i] = bus2.kVBase
            nphases[i] = nodes.size
            if nodes.size > 3: nodes = nodes[0:3]
            cidx = 2 * np.array(range(0, min(int(v.size / 2), 3)))
            bus1 = self.dssCircuit.Buses(re.sub(r"\..*", "", el.BusNames[0]))
            busname[i] = bus1.Name
            src.append(busname[i])
            dest.append(busnameto[i])
            src_x.append(bus1.x)
            src_y.append(bus1.y)
            dst_x.append(bus2.x)
            dst_y.append(bus2.y)
            Vto[i, nodes - 1] = v[cidx] + 1j * v[cidx + 1]
            x[i] = bus1.x
            y[i] = bus1.y
            v = np.array(bus1.Voltages)
            V[i, nodes - 1] = v[cidx] + 1j * v[cidx + 1]

            current = np.array(el.Currents)

            # to deal with certain exceptions
            if ('SwtControl' in el.Name or 'RegControl' in el.Name) and current.shape[0] / 2 < cidx.shape[0]:
                continue
            I[i, nodes - 1] = current[cidx] + 1j * current[cidx + 1]
            pflows[i, nodes - 1] = (V[i, nodes - 1] * I[i, nodes - 1].conj()).real / 1000
            qflows[i, nodes - 1] = (V[i, nodes - 1] * I[i, nodes - 1].conj()).imag / 1000
            i = i + 1

        nodes = []
        xcoords = []
        ycoords = []
        for i, s in enumerate(src):
            if s not in nodes:
                nodes.append(s)
                xcoords.append(src_x[i])
                ycoords.append(src_y[i])
        for j, d in enumerate(dest):
            if d not in nodes:
                nodes.append(d)
                xcoords.append(dst_x[j])
                ycoords.append(dst_y[j])

        G = nx.Graph()

        pos = {}
        N = self.dssCircuit.NumBuses
        for i in range(N):
            bus = self.dssCircuit.Buses(i)
            pos[bus.Name] = (bus.x, bus.y)

       # certain locations were not available hence, some conditions are set
        for i, n in enumerate(nodes):
            if str(n) in pos.keys():
                if self.case == 'hce' and n == 'sm451b':
                    if str(n) in node_type.keys():
                        G.add_node(n, pos=(2600000.00, 1510000.00), nodetype=node_type[str(n)])
                    else:
                        G.add_node(n, pos=(2600000.00, 1510000.00), nodetype='Node')
                else:
                    if str(n) in node_type.keys():
                        G.add_node(n, pos=pos[str(n)], nodetype=node_type[str(n)])
                    else:
                        G.add_node(n, pos=pos[str(n)], nodetype='Node')
            else:
                if self.case == 'hce':
                    if str(n) in node_type.keys():
                        G.add_node(n, pos=(2600000.00, 1510000.00), nodetype=node_type[str(n)])
                    else:
                        G.add_node(n, pos=(2600000.00, 1510000.00), nodetype='Node')
                else:
                    if str(n) in node_type.keys():
                        G.add_node(n, pos=(0.0, 0.0), nodetype=node_type[str(n)])
                    else:
                        G.add_node(n, pos=(0.0, 0.0), nodetype='Node')

        pflow_list = []
        for ix, pf in enumerate(pflows):
            pflow_list.append(abs(np.average(pflows[ix])))

        for s, d, pf in zip(src, dest, pflow_list):
            if s != d:
                G.add_edge(s, d, weights=pf, pflow=pf)


        edge_attribs = G.edges(data=True)

        Adj = np.zeros((len(nodes), len(nodes)))

        if self.case not in ['hce','hce_mod']:
            for s in list(G.nodes):
                for d in list(G.nodes):
                    if s == d:
                        Adj[list(G.nodes).index(s)][list(G.nodes).index(d)] = 0
                    else:
                        match = [item[2]['weights'] for item in edge_attribs if (
                                    (str(item[0]) == s and str(item[1]) == d) or (
                                        str(item[0]) == d and str(item[1]) == s))]
                        if len(match) > 0:
                            Adj[list(G.nodes).index(s)][list(G.nodes).index(d)] = match[0]
                        else:
                            Adj[list(G.nodes).index(s)][list(G.nodes).index(d)] = self.M

        power_data = pd.read_csv(self.power_file)
        # extract load rows
        element_list = list(power_data['Element'].values)
        load_ix = []
        gen_ix=[]
        for ix, item in enumerate(element_list):
            if 'Load' in item:
                load_ix.append(ix)
            elif 'Generator' in item:
                gen_ix.append(ix)
        load_df = power_data.loc[load_ix]
        gen_df = power_data.loc[gen_ix]
        for k, v in load_info.items():
            mask = load_df['Element'].values == v
            load_info[k] = (v, load_df.iloc[mask][' P_1'].values[0])
        for k, v in gen_info.items():
            mask = gen_df['Element'].values == v
            gen_info[k] = (v, gen_df.iloc[mask][' P_1'].values[0])

        return G, Adj, xcoords, ycoords, load_info,gen_info