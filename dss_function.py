# ~~~~~~~~~~~~ Author: Fei Ding @ NREL ~~~~~~~~~~~~~~~

import numpy as np
import math
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import networkx as nx
import random
import os
import pandas as pd
import csv


def get_lines(dss):
    Line = []
    Switch = []
    lines = dss.Lines.First()
    while lines:
        datum = {}
        line = dss.Lines
        datum["name"] = line.Name()
        datum["bus1"] = line.Bus1()
        datum["bus2"] = line.Bus2()
        datum["switch_flag"] = dss.run_command('? Line.' + datum["name"] + '.Switch')
        if datum["switch_flag"] == 'False':
            datum["wires"] = dss.run_command('? Line.' + datum["name"] + '.Wires')
            datum["length"] = line.Length()
            datum['units'] = line.Units()
            datum["phases"] = line.Phases()
            datum["spacing"] = line.Spacing()
            datum["linecode"] = line.LineCode()
            datum["normAmp"] = line.NormAmps()
            datum["geometry"] = line.Geometry()
            #datum["R1"] = line.R1()*line.Length()
            #datum["X1"] = line.X1()*line.Length()
            datum["RMatrix"] = [x*line.Length() for x in line.RMatrix()]
            datum["XMatrix"] = [x*line.Length() for x in line.XMatrix()]
            Line.append(datum)
        else:
            Switch.append(datum)
        lines = dss.Lines.Next()
    return [Line, Switch]


def get_transformer(dss, circuit):
    data = []
    circuit.SetActiveClass('Transformer')
    xfmr_index = dss.ActiveClass.First()
    while xfmr_index:
        cktElement = dss.CktElement
        xfmr_name = cktElement.Name()
        buses = cktElement.BusNames()
        conns = dss.run_command('? ' + xfmr_name + '.conns')
        kVs = dss.run_command('? ' + xfmr_name + '.kVs')
        kVAs = dss.run_command('? ' + xfmr_name + '.kVAs')
        phase = dss.run_command('? ' + xfmr_name + '.phases')
        loadloss = dss.run_command('? ' + xfmr_name + '.%loadloss')
        noloadloss = dss.run_command('? ' + xfmr_name + '.%noloadloss')
        Rs = dss.run_command('? ' + xfmr_name + '.%Rs')
        xhl = dss.run_command('? ' + xfmr_name + '.xhl')
        dataline = dict(name=xfmr_name, buses=buses, conns=conns, kVs=kVs, kVAs=kVAs, phase=phase, loadloss=loadloss,
                        noloadloss=noloadloss, Rs=Rs, xhl=xhl)
        data.append(dataline)
        xfmr_index = dss.ActiveClass.Next()
    return data


def get_element(dss, circuit, drop_element_name=None):
    Element = ['Line.' + ii for ii in dss.Lines.AllNames()] + ['Transformer.' + ii for ii in dss.Transformers.AllNames()]
    if drop_element_name:
        for elem in drop_element_name:
            Element.remove(elem)
    Element_index = []  # this is the index to map AllElement_noPhase into AllElement_name
    AllElement_capacity = []
    AllElement_name = []
    numElement = 0
    for elem in Element:
        if elem == 'Line.sw1':
            print(1)
        temp = []
        circuit.SetActiveElement(elem)
        step = int(len(dss.CktElement.NodeOrder()) / 2)
        numElement = numElement + step
        count = 0
        for ii in range(step):
            AllElement_capacity.append(dss.CktElement.NormalAmps())
            AllElement_name.append(elem + '.' + str(dss.CktElement.NodeOrder()[ii]))
            temp.append(count)
            count = count + 1
        Element_index.append(temp)
    resstring = np.column_stack((AllElement_name, AllElement_capacity))
    resfile = os.path.join('element_capacity.csv')
    fn = open(resfile, 'w')
    np.savetxt(fn, resstring, fmt='%s', delimiter=',')
    fn.close()
    return [Element, AllElement_name, AllElement_capacity, numElement, Element_index]


def get_loads(dss, circuit, loadshape_flag=0, loadshape_folder=None):
    data = []
    load_flag = dss.Loads.First()
    total_load = 0

    # if loadshape_flag == 1:
    #     filename = os.path.join(loadshape_folder, 'Load_shape_1sec_1day.csv')
    #     data0 = []
    #     with open(filename, 'r') as f:
    #         csvread = csv.reader(f)
    #         for row in csvread:
    #             data0.append(float(row[0]))
    #         f.close()

    while load_flag:
        load = dss.Loads
        datum = {
            "name": load.Name(),
            "kV": load.kV(),
            "kW": load.kW(),
            "PF": load.PF(),
            "Delta_conn": load.IsDelta()
        }
        indexCktElement = circuit.SetActiveElement("Load.%s" % datum["name"])
        cktElement = dss.CktElement
        bus = cktElement.BusNames()[0].split(".")
        datum["kVar"] = float(datum["kW"]) / float(datum["PF"]) * math.sqrt(1 - float(datum["PF"]) * float(datum["PF"]))
        datum["bus1"] = bus[0]
        datum["numPhases"] = len(bus[1:])
        datum["phases"] = bus[1:]
        if not datum["numPhases"]:
            datum["numPhases"] = 3
            datum["phases"] = ['1', '2', '3']
        datum["voltageMag"] = cktElement.VoltagesMagAng()[0]
        datum["voltageAng"] = cktElement.VoltagesMagAng()[1]
        datum["power"] = dss.CktElement.Powers()[0:2]

        data.append(datum)
        load_flag = dss.Loads.Next()
        total_load += datum["kW"]

        if loadshape_flag == 1:  # read loadshape file to get 1-year data
            filename = os.path.join(loadshape_folder, datum["name"] + '_loadshape.csv')
            data0 = []
            with open(filename, 'r') as f:
                csvread = csv.reader(f)
                for row in csvread:
                    data0.append(float(row[0]))
                f.close()
            datum["1year_loadshape"] = data0

    return [data, total_load]


def get_baseload(dss):
    data = []
    load_flag = dss.Loads.First()
    while load_flag:
        datum = {}
        Loadname = dss.CktElement.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        GENkW = dss.run_command('? ' + Loadname + '.kW')
        GENpf = dss.run_command('? ' + Loadname + '.pf')
        GENkVA = dss.run_command('? ' + Loadname + '.kVA')
        GENkV = dss.run_command('? ' + Loadname + '.kV')
        datum["name"] = Loadname
        datum["bus"] = bus
        datum["kW"] = GENkW
        datum["pf"] = GENpf
        datum["kV"] = GENkV
        datum["kVA"] = GENkVA
        datum["numPhase"] = NumPhase
        data.append(datum)
        load_flag = dss.Loads.Next()
    return data


def generate_PV(Load, totalLoadkW, target_penetration, kW_kVA_ratio, outputfile):
    # need to use "get_loads" function to get "Load" and "totalLoadkW"
    pv_power = 0
    count = 1
    candidate = np.array(range(len(Load))).tolist()
    pv_dss = []
    while pv_power <= target_penetration / 100 * totalLoadkW:
        if not candidate:
            candidate = np.array(range(len(Load))).tolist()
        script = []
        load_index = random.randint(0, len(candidate) - 1)
        load1 = Load[candidate[load_index]]

        for phase_no in load1["phases"]:
            pvname = load1["name"] + '_PV' + str(count)
            busname = load1["bus1"] + '.' + str(phase_no)
            kW = round(random.uniform(0, 2 * load1["kW"]), 2)  # randomly generate single PV system
            # kW = round(random.uniform(0,200),2)  # generate aggreated PVs
            if load1["Delta_conn"] == 0:
                conn = 'wye'
            else:
                conn = 'delta'
            script = 'New Generator.' + pvname + ' bus1=' + busname + ' conn=' + conn + ' phases=1 kV=' + str(
                load1["kV"] / math.sqrt(load1["numPhases"])) + ' kW=' + str(kW) + ' kVA=' + str(
                kW * kW_kVA_ratio) + ' pf=1 !yearly=PVshape_aggregated'
            count = count + 1
            pv_dss.append(script)
            pv_power = pv_power + kW
        candidate.remove(candidate[load_index])

    file = open(outputfile, 'w')
    for string in pv_dss:
        file.write(string + '\n')
    file.close()


def get_pvSystems(dss):
    data = []
    PV_flag = dss.PVsystems.First()
    while PV_flag:
        datum = {}
        PVname = dss.CktElement.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]

        PVkW = dss.run_command('? ' + PVname + '.Pmpp')
        PVpf = dss.run_command('? ' + PVname + '.pf')
        PVkVA = dss.run_command('? ' + PVname + '.kVA')
        PVkV = dss.run_command('? ' + PVname + '.kV')

        datum["name"] = PVname
        datum["bus"] = bus
        datum["Pmpp"] = PVkW
        datum["pf"] = PVpf
        datum["kV"] = PVkV
        datum["kVA"] = PVkVA
        datum["numPhase"] = NumPhase
        datum["power"] = dss.CktElement.Powers()[0:2 * NumPhase]

        data.append(datum)
        PV_flag = dss.PVsystems.Next()
    return data


def get_Generator(dss):
    data = []
    gen_flag = dss.Generators.First()
    while gen_flag:
        datum = {}
        GENname = dss.CktElement.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        GENkW = dss.run_command('? ' + GENname + '.kW')
        GENpf = dss.run_command('? ' + GENname + '.pf')
        GENkVA = dss.run_command('? ' + GENname + '.kVA')
        GENkV = dss.run_command('? ' + GENname + '.kV')
        datum["name"] = GENname
        datum["bus"] = bus
        datum["kW"] = GENkW
        datum["pf"] = GENpf
        datum["kV"] = GENkV
        datum["kVA"] = GENkVA
        datum["numPhase"] = NumPhase
        # datum["power"] = dss.CktElement.Powers()[0:2*NumPhase]
        data.append(datum)
        gen_flag = dss.Generators.Next()
    return data


def get_Storage(dss):
    data_storages = []
    storage_dataframe = dss.utils.class_to_dataframe('Storage')
    for strg_ in storage_dataframe.index:
        strg_name = strg_
        kWhRated = float(float(storage_dataframe['kWhrated'][strg_]))
        kWRated = float(float(storage_dataframe['kWrated'][strg_]))
        buses = storage_dataframe['bus1'][strg_]
        kVs = float(storage_dataframe['kv'][strg_])
        kVAs = float(storage_dataframe['kVA'][strg_])
        phase = int(storage_dataframe['phases'][strg_])
        percent_stored = float(storage_dataframe['%stored'][strg_])
        kWh_stored = float(storage_dataframe['kWhstored'][strg_])
        efficiency_charge = float(storage_dataframe["%EffCharge"][strg_])
        efficiency_discharge = float(storage_dataframe["%EffDischarge"][strg_])
        dataline = dict(name=strg_name, bus=buses, kV=kVs, kVA=kVAs, kWrated=kWRated, kWhrated=kWhRated, numPhase=phase,
                        prev_kWh_stored=kWh_stored,
                        batt_soc=percent_stored, efficiency_charge=efficiency_charge,
                        efficiency_discharge=efficiency_discharge,
                        mu_SOC_upper=0, mu_SOC_lower=0)
        data_storages.append(dataline)
    return data_storages


def get_capacitors(dss):
    data = []
    cap_flag = dss.Capacitors.First()
    while cap_flag:
        datum = {}
        capname = dss.CktElement.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        kvar = dss.run_command('? ' + capname + '.kVar')
        datum["name"] = capname
        temp = bus.split('.')
        datum["busname"] = temp[0]
        datum["busphase"] = temp[1:]
        if not datum["busphase"]:
            datum["busphase"] = ['1', '2', '3']
        datum["kVar"] = kvar
        datum["numPhase"] = NumPhase
        datum["power"] = dss.CktElement.Powers()[0:2 * NumPhase]
        data.append(datum)
        cap_flag = dss.Capacitors.Next()
    return data


def get_BusDistance(dss, circuit, AllNodeNames):
    Bus_Distance = []
    for node in AllNodeNames:
        circuit.SetActiveBus(node)
        Bus_Distance.append(dss.Bus.Distance())
    return Bus_Distance


def get_PVmaxP(dss, circuit, PVsystem):
    Pmax = []
    for PV in PVsystem:
        circuit.SetActiveElement(PV["name"])
        Pmax.append(-float(dss.CktElement.Powers()[0]))
    return Pmax


def get_PQnode(dss, circuit, Load, PVsystem, AllNodeNames, Capacitors):
    Pload = [0] * len(AllNodeNames)
    Qload = [0] * len(AllNodeNames)
    for ld in Load:
        for ii in range(len(ld['phases'])):
            name = ld['bus1'] + '.' + ld['phases'][ii]
            index = AllNodeNames.index(name.upper())
            circuit.SetActiveElement('Load.' + ld["name"])
            power = dss.CktElement.Powers()
            Pload[index] = power[2 * ii]
            Qload[index] = power[2 * ii + 1]
            # Pload[index] = ld['kW']/ld['numPhases']
            # Qload[index] = ld['kVar'] / ld['numPhases']

    # PQ_load = np.matrix(np.array(Pload) + 1j * np.array(Qload)).transpose()
    PQ_load = np.array(Pload) + 1j * np.array(Qload)

    Ppv = [0] * len(AllNodeNames)
    Qpv = [0] * len(AllNodeNames)
    for PV in PVsystem:
        bus = PV["bus"].split('.')
        if len(bus) == 1:
            bus = bus + ['1', '2', '3']
        circuit.SetActiveElement(PV["name"])
        power = dss.CktElement.Powers()
        for ii in range(len(bus) - 1):
            index = AllNodeNames.index((bus[0] + '.' + bus[ii + 1]).upper())
            Ppv[index] = power[2 * ii]
            Qpv[index] = power[2 * ii + 1]

    # PQ_PV = np.matrix(np.array(Ppv) + 1j * np.array(Qpv)).transpose()
    PQ_PV = -np.array(Ppv) - 1j * np.array(Qpv)

    Qcap = [0] * len(AllNodeNames)
    for cap in Capacitors:
        for ii in range(cap["numPhase"]):
            index = AllNodeNames.index(cap["busname"].upper() + '.' + cap["busphase"][ii])
            Qcap[index] = -cap["power"][2 * ii - 1]

    PQ_node = - PQ_load + PQ_PV + 1j * np.array(Qcap)  # power injection
    return [PQ_load, PQ_PV, PQ_node, Qcap]


def get_subPower_byPhase(dss):
    dss.Lines.First()
    power = dss.CktElement.Powers()
    subpower = power[0:6:2]
    return subpower


def getPowers(circuit, dss, type, names):
    d = [None] * len(names)
    count = 0
    for loadname in names:
        if len(loadname.split('.')) > 1:
            circuit.SetActiveElement(loadname)
        else:
            circuit.SetActiveElement(type + '.' + loadname)
        s = dss.CktElement.Powers()
        d[count] = [sum(s[0:len(s):2]), sum(s[1:len(s):2])]
        count = count + 1
    d = np.asarray(d)
    powers = sum(d)
    return [d, powers]


def construct_Ymatrix(Ysparse, slack_no, totalnode_number, order_number):
    Ymatrix = np.array([[complex(0, 0)] * totalnode_number] * totalnode_number)
    file = open(Ysparse, 'r')
    G = []
    B = []
    count = 0
    for line in file:
        if count >= 4:
            temp = line.split('=')
            temp_order = temp[0]
            temp_value = temp[1]
            temp1 = temp_order.split(',')
            row_value = int(temp1[0].replace("[", ""))
            column_value = int(temp1[1].replace("]", ""))
            row_value = order_number[row_value - 1]
            column_value = order_number[column_value - 1]
            temp2 = temp_value.split('+')
            G.append(float(temp2[0]))
            B.append(float(temp2[1].replace("j", "")))
            Ymatrix[row_value, column_value] = complex(G[-1], B[-1])
            Ymatrix[column_value, row_value] = complex(G[-1], B[-1])
        count = count + 1
    file.close()

    Y00 = Ymatrix[0:slack_no, 0:slack_no]
    Y01 = Ymatrix[0:slack_no, slack_no:]
    Y10 = Ymatrix[slack_no:, 0:slack_no]
    Y11 = Ymatrix[slack_no:, slack_no:]
    Y11_sparse = lil_matrix(Y11)
    Y11_sparse = Y11_sparse.tocsr()
    a_sps = sparse.csc_matrix(Y11)
    lu_obj = sp.splu(a_sps)
    Y11_inv = lu_obj.solve(np.eye(totalnode_number - slack_no))
    return [Y00, Y01, Y10, Y11, Y11_sparse, Y11_inv, Ymatrix]


def re_orgnaize_for_volt(V1_temp, AllNodeNames, NewNodeNames):
    V1 = [complex(0, 0)] * len(V1_temp)
    count = 0
    for node in NewNodeNames:
        index = AllNodeNames.index(node)
        print([count, index])
        V1[index] = V1_temp[count]
        count = count + 1
    return V1


def getCapsPos(dss, capNames):
    o = [None] * len(capNames)
    for i, cap in enumerate(capNames):
        x = dss.run_command('? capacitor.%(cap)s.states' % locals())
        o[i] = int(x[-2:-1])
    return o


def getRegsTap(dss, regNames):
    o = [None] * len(regNames)
    for i, name in enumerate(regNames):
        xfmr = dss.run_command('? regcontrol.%(name)s.transformer' % locals())
        res = dss.run_command('? transformer.%(xfmr)s.tap' % locals())
        o[i] = float(res)
    return o


def result(circuit, dss):
    res = {}
    res['AllVoltage'] = circuit.AllBusMagPu()
    temp = circuit.YNodeVArray()
    data = []
    for ii in range(int(len(temp) / 2)):
        data.append(complex(temp[2 * ii], temp[2 * ii + 1]))
    res['AllVolt_Yorder'] = data
    res['loss'] = circuit.Losses()
    res['totalPower'] = circuit.TotalPower()  # power generated into the circuit
    loadname = dss.Loads.AllNames()
    res['totalLoadPower'] = getPowers(circuit, dss, 'Load', loadname)[1]
    capNames = dss.Capacitors.AllNames()
    if capNames:
        res['CapState'] = getCapsPos(dss, capNames)
    else:
        res['CapState'] = 'nan'
    regNames = dss.RegControls.AllNames()
    if regNames:
        res['RegTap'] = getRegsTap(dss, regNames)
    else:
        res['RegTap'] = 'nan'

    pvNames = dss.Generators.AllNames()
    dataP = np.zeros(len(pvNames))
    dataQ = np.zeros(len(pvNames))
    ii = 0
    sumP = 0
    sumQ = 0
    for pv in pvNames:
        #        circuit.SetActiveElement('PVsystem.'+pv)
        circuit.SetActiveElement('Generator.' + pv)
        tempPQ = dss.CktElement.Powers()
        dataP[ii] = sum(tempPQ[0:len(tempPQ):2])
        dataQ[ii] = sum(tempPQ[1:len(tempPQ):2])
        sumP = sumP + dataP[ii]
        sumQ = sumQ + dataQ[ii]
        ii = ii + 1
    res['PV_Poutput'] = dataP
    res['PV_Qoutput'] = dataQ
    res['totalPVpower'] = [sumP, sumQ]
    return res


def get_Vbus(dss, circuit, busname):  # busname doesn't has .1, .2, or .3
    circuit.SetActiveBus(busname)
    voltage = dss.Bus.VMagAngle()
    Vmag = [ii / dss.Bus.kVBase() / 1000 for ii in voltage[0:len(voltage):2]]
    # Vmag = [ii/1 for ii in voltage[0:len(voltage):2]]
    return Vmag


def get_Vnode(dss, circuit, nodename):  # this one should only return 1 value/node
    circuit.SetActiveBus(nodename)
    voltage = dss.Bus.VMagAngle()
    Vmag = [ii / dss.Bus.kVBase() / 1000 for ii in voltage[0:len(voltage):2]]
    allbusnode = dss.Bus.Nodes()
    phase = nodename.split('.')[1]
    index = allbusnode.index(int(phase))
    Vnode = Vmag[index]
    return Vnode


def get_voltage_Yorder(circuit, node_number, Vbase):
    temp_Vbus = circuit.YNodeVArray()
    voltage = [complex(0, 0)] * node_number
    for ii in range(node_number):
        voltage[ii] = complex(temp_Vbus[ii * 2], temp_Vbus[ii * 2 + 1])
    voltage_pu = list(map(lambda x: abs(x[0]) / x[1], zip(voltage, Vbase)))
    return [voltage, voltage_pu]


def create_graph(dss, phase):
    df = dss.utils.lines_to_dataframe()
    G = nx.Graph()
    data = df[['Bus1', 'Bus2']].to_dict(orient="index")
    for name in data:
        line = data[name]
        if ".%s" % phase in line["Bus1"] and ".%s" % phase in line["Bus2"]:
            G.add_edge(line["Bus1"].split(".")[0], line["Bus2"].split(".")[0])
    pos = {}
    for name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus("%s" % name)
        if phase in dss.Bus.Nodes():
            index = dss.Bus.Nodes().index(phase)
            re, im = dss.Bus.PuVoltage()[2 * index:2 * index + 2]
            V = abs(complex(re, im))
            D = dss.Bus.Distance()
            pos[dss.Bus.Name()] = (D, V)
    return G, pos


def plot_profile(dss, phase):
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    ax = axs
    ncolor = ['k', 'r', 'b']
    nshape = ['o', '+', '*']
    plt.rcParams.update({'font.size': 20})
    if phase == 1:
        G, pos = create_graph(dss, 1)
        nx.draw_networkx_nodes(G, pos, ax=ax, with_labels=False, node_size=16, node_color=ncolor[phase - 1])
        nx.draw_networkx_edges(G, pos, ax=ax, with_labels=False, node_size=16, node_color=ncolor[phase - 1])
        ax.set_title("Voltage profile plot for phase A")

    elif phase == 2:
        G, pos = create_graph(dss, 2)
        nx.draw_networkx_nodes(G, pos, ax=ax, with_labels=False, node_size=16, node_color=ncolor[phase - 1])
        nx.draw_networkx_edges(G, pos, ax=ax, with_labels=False, node_size=16, node_color=ncolor[phase - 1])
        ax.set_title("Voltage profile plot for phase B")
    elif phase == 3:
        G, pos = create_graph(dss, 3)
        nx.draw_networkx_nodes(G, pos, ax=ax, with_labels=False, node_size=16, node_color=ncolor[phase - 1])
        nx.draw_networkx_edges(G, pos, ax=ax, with_labels=False, node_size=16, node_color=ncolor[phase - 1])
        ax.set_title("Voltage profile plot for phase C")
    else:
        for ph in range(3):
            G, pos = create_graph(dss, ph + 1)
            nx.draw_networkx_nodes(G, pos, ax=ax, with_labels=False, node_size=40, node_color=ncolor[ph],
                                   node_shape=nshape[ph])
            # nx.draw_networkx_edges(G, pos, ax=ax, with_labels=False, node_size=16, node_color=ncolor[ph])
        ax.set_title("Voltage profile plot for all phases")
        ax.legend(['phase a', 'phase b', 'phase c'])

    ax.grid()
    ax.set_ylabel("Voltage in p.u.")
    ax.set_xlabel("Distances in km")
    plt.show()


def engo_setpoint_update(dss, circuit, dfEngos, setpoint, engo_fname):
    engoname = []
    for engonum in range(len(dfEngos)):
        engoname.append('ENGO' + str(engonum + 1))

    f = open(engo_fname, 'w')
    for engo in range(len(dfEngos)):
        # for engo in range(3):
        f.write('New Capacitor.%s Phases=1 Bus1=%s kvar=10 numsteps=10 kv=%s\n' % (
        engoname[engo], dfEngos.loc[engo, 'bus'], str(round(dfEngos.loc[engo, 'kV'], 3))))
        f.write('New capcontrol.%s Element=Transformer.%s Terminal=2 capacitor=%s ctr=1 ptr=1  EventLog=Yes \n\
                 ~ usermodel="C:\Program Files\OpenDSS\\x64\ENGOCapControl_12345_sec.dll" \n\
                 ~ userdata=(ENABLE=Y Vnom=%s Vsp_120b =%s Vband_120b =1 )\n\n' % (
        dfEngos.loc[engo, 'Name'], dfEngos.loc[engo, 'Transformer'], dfEngos.loc[engo, 'Name'],
        str(int(float(dfEngos.loc[engo, 'kV']) * 1000)), str(setpoint)))
    f.close()


def getElemCurrents(circuit, dss, type, name):
    circuit.SetActiveElement(type + '.' + name)
    s = dss.CktElement.CurrentsMagAng()
    magIbyPh = s[0:len(s):2][:3]
    return magIbyPh


def get3phLinePower(circuit, dss, type, name):
    circuit.SetActiveElement(type + '.' + name)
    s = dss.CktElement.Powers()
    lens = int(len(s) / 2)
    powers = [sum(s[0:lens:2]), sum(s[1:lens:2])]
    return powers


def get_phase_impedance_matrix(numElement, AllElement, dss, circuit, AllNodeNames):
    def list2matrix(xx):
        size = int(math.sqrt(len(xx) / 2))
        yy = np.array([[complex(0, 0)] * size] * size)
        for ii in range(size):
            for jj in range(size):
                yy[ii, jj] = complex(xx[2 * jj + size * 2 * ii], xx[2 * jj + 1 + size * 2 * ii])
        return yy

    Zbranch = np.array([[complex(0, 0)] * numElement] * numElement)
    branch_node_incidence = np.zeros([numElement, len(AllNodeNames)])
    count = 0
    for elem in AllElement:
        # if elem.split('.')[0].lower() == 'transformer':
        #     print(1)
        circuit.SetActiveElement(elem)
        buses = [ii.split('.')[0] for ii in dss.CktElement.BusNames()]
        nodes = dss.CktElement.NodeOrder()
        while 1:
            if 0 in nodes:
                nodes.remove(0)
            else:
                break
        numPhase = dss.CktElement.NumPhases()
        circuit.SetActiveBus(buses[0])
        kV1 = dss.Bus.kVBase()
        tempZsc1 = list2matrix(dss.Bus.ZscMatrix())
        if not np.shape(tempZsc1)[0] == numPhase:
            allnodes = dss.Bus.Nodes()
            existnodes = nodes[0:int(len(nodes) / 2)]
            drop_index = list(set(allnodes) - set(existnodes))
            if len(drop_index) == 1:
                Zsc1 = np.delete(tempZsc1, (allnodes.index(drop_index[0])), axis=0)
                Zsc1 = np.delete(Zsc1, (allnodes.index(drop_index[0])), axis=1)
            elif len(drop_index) == 2:
                existnodes_index = allnodes.index(existnodes[0])
                Zsc1 = tempZsc1[existnodes_index, existnodes_index]
        else:
            Zsc1 = tempZsc1
        circuit.SetActiveBus(buses[1])
        kV2 = dss.Bus.kVBase()
        tempZsc2 = list2matrix(dss.Bus.ZscMatrix())
        if not np.shape(tempZsc2)[0] == numPhase:
            allnodes = dss.Bus.Nodes()
            existnodes = nodes[0:int(len(nodes) / 2)]
            drop_index = list(set(allnodes) - set(existnodes))
            if len(drop_index) == 1:
                Zsc2 = np.delete(tempZsc2, (allnodes.index(drop_index[0])), axis=0)
                Zsc2 = np.delete(Zsc2, (allnodes.index(drop_index[0])), axis=1)
            elif len(drop_index) == 2:
                existnodes_index = allnodes.index(existnodes[0])
                Zsc2 = tempZsc2[existnodes_index, existnodes_index]
        else:
            Zsc2 = tempZsc2
        start_no = count
        Zelem = (Zsc2 - Zsc1) * (kV1 / kV2) * (kV1 / kV2)
        for ii in range(numPhase):
            from_node = buses[0] + '.' + str(nodes[ii])
            to_node = buses[1] + '.' + str(nodes[ii + numPhase])
            from_node_index = AllNodeNames.index(from_node.upper())
            to_node_index = AllNodeNames.index(to_node.upper())
            branch_node_incidence[count, from_node_index] = 1
            branch_node_incidence[count, to_node_index] = -1 * kV1 / kV2
            count = count + 1
        end_no = count
        Zbranch[start_no:end_no, start_no:end_no] = Zelem
    return [Zbranch, branch_node_incidence]


def generate_AddMarker(PV_location, outputfile, code, color, size):
    file = open(outputfile, 'w')
    for bus in PV_location:
        string = 'AddBusMarker Bus=' + bus.split('.')[0] + ' code=' + code + ' color=' + color + ' size=' + size
        file.write(str(string) + '\n')
    file.close()


def reform_dssCurrent_export(filename, num_Element, Element, Element_index):
    Idata = pd.read_csv(filename)
    Ibranch_export = [0] * num_Element
    allname = Idata.loc[:, "Element"]
    count = 0
    jj = 0
    for name in Element:
        ii = list(allname.values).index(name.split('.')[0] + '.' + name.split('.')[1].upper())
        temp = Idata.iloc[ii, [1, 3, 5, 7, 11, 13, 15, 17]]
        elementname_index = Element_index[count]
        for kk in elementname_index:
            Ibranch_export[jj] = temp[kk]
            jj = jj + 1
        count = count + 1
    return Ibranch_export


def convert_3phasePV_to_1phasePV(PVSystem_3phase):
    # convert multi-phase PV into multiple 1-phase PVs for control implementation purpose
    PVSystem_1phase = []
    multiphase_PVname = []
    for pv in PVSystem_3phase:
        bus = pv["bus"].split('.')
        if len(bus) == 1:
            bus = bus + ['1', '2', '3']
            multiphase_PVname.append(pv["name"])
        for ii in range(int(pv["numPhase"])):
            pv_perphase = {}
            pv_perphase["pf"] = pv["pf"]
            pv_perphase["kV"] = pv["kV"]
            pv_perphase["numPhase"] = pv["numPhase"]
            pv_perphase["name"] = pv["name"]  # +'_node'+bus[ii+1]
            pv_perphase["bus"] = bus[0] + '.' + bus[ii + 1]
            pv_perphase["kW"] = str(float(pv["kW"]) / float(pv["numPhase"]))
            pv_perphase["kVA"] = str(float(pv["kVA"]) / float(pv["numPhase"]))
            PVSystem_1phase.append(pv_perphase)
    return [PVSystem_1phase,multiphase_PVname]


def get_Ymatrix(dss, circuit, feeder_directory):
    # ------- get Ymatrix file ------------
    dss.run_command('vsource.source.enabled=no')
    # dss.run_command('Disable RegControl..*')
    dss.run_command('solve')
    dss.run_command('show Y')
    YNodeNames = circuit.YNodeOrder()
    YNodeNames = [str(node) for node in YNodeNames]
    with open(os.path.join(feeder_directory, 'result_nodename.csv'), 'w') as f:
        csvwriter = csv.writer(f)
        for ii in range(len(YNodeNames)):
            csvwriter.writerow([YNodeNames[ii]])
    f.close()
    dss.run_command('vsource.source.enabled=yes')


def get_Yprim_matrix(numElement, AllElement, dss, circuit, AllNodeNames):
    def form_Yprim(values):
        dimension = int(math.sqrt(len(values) / 2))
        Yprim = np.array([[complex(0, 0)] * dimension] * dimension)
        for ii in range(dimension):
            for jj in range(dimension):
                Yprim[ii][jj] = complex(values[dimension * ii * 2 + 2 * jj], values[dimension * ii * 2 + 2 * jj + 1])
        return Yprim

    Ybranch_prim = np.array([[complex(0, 0)] * 2 * numElement] * 2 * numElement)
    branch_node_incidence = np.zeros([2 * numElement, len(AllNodeNames) + 966])  # 966 for HCE feeder
    temp_AllNodeNames = [ii for ii in AllNodeNames]
    count = 0
    neutral_count = 0
    start_no = 0
    record_index = []
    for elem in AllElement:
        circuit.SetActiveElement(elem)
        values = dss.CktElement.YPrim()
        Yprim = form_Yprim(values)
        buses = [ii.split('.')[0] for ii in dss.CktElement.BusNames()]
        nodes = dss.CktElement.NodeOrder()
        numPhase = int(len(nodes) / 2)
        for ii in range(numPhase):
            from_node = buses[0] + '.' + str(nodes[ii])
            to_node = buses[1] + '.' + str(nodes[ii + numPhase])
            if nodes[ii] == 0:
                temp_AllNodeNames.append(from_node.upper())
                neutral_count = neutral_count + 1
            if nodes[ii + numPhase] == 0:
                temp_AllNodeNames.append(to_node.upper())
                neutral_count = neutral_count + 1
            from_node_index = temp_AllNodeNames.index(from_node.upper())
            to_node_index = temp_AllNodeNames.index(to_node.upper())
            branch_node_incidence[2 * count + ii, from_node_index] = 1
            branch_node_incidence[2 * count + numPhase + ii, to_node_index] = 1
            record_index.append(2 * count + ii)
        count = count + numPhase
        end_no = start_no + 2 * numPhase
        Ybranch_prim[start_no:end_no, start_no:end_no] = Yprim
        start_no = end_no
    return [Ybranch_prim, branch_node_incidence, neutral_count, record_index]


def system_topology_matrix_form(dss, circuit, AllNodeNames, slack_number, Ynode_file, drop_element_name):
    node_number = len(AllNodeNames)
    YNodeNames = []
    with open(Ynode_file, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            YNodeNames.append(row[0])
    f.close()
    # get the corresponding order of nodes:
    order_nodes = []
    for node in YNodeNames:
        order_nodes.append(AllNodeNames.index(node))
    Ysparse_file = os.path.join(os.getcwd(), str(circuit.Name()).upper() + '_SystemY.txt')
    Yinformation = construct_Ymatrix(Ysparse_file, slack_number, node_number,
                                     order_nodes)  # Yinformation = [Y00, Y01, Y10, Y11, Y11_sparse, Y11_inv, Ymatrix]
    print('Finish getting Ybus information')
    # ------- get node branch incidence matrix ------------
    dss.run_command('solve mode=fault')
    Element_information = get_element(dss, circuit,
                                      drop_element_name)  # Element_information = [Element, AllElement_name, AllElement_Capacity, numElement, Element_index]
    current_coeff_matrix = []
    record_index = []
    # [Ybranch_prim, branch_node_incidence, neutral_count, record_index] = get_Yprim_matrix(Element_information[3],
    #                                                                                                    Element_information[0], dss,
    #                                                                                                    circuit,
    #                                                                                                    AllNodeNames)
    # current_coeff_matrix = np.dot(Ybranch_prim, branch_node_incidence)
    # current_coeff_matrix = current_coeff_matrix[record_index, :-neutral_count]
    # print('Finish getting Zbranch information')
    return [Yinformation, current_coeff_matrix, Element_information, record_index]

# def auto_voltVAR_voltWatt():
#
#     return [Pinv,Qinv]
