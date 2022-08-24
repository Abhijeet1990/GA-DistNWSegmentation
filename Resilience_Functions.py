import numpy as np


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

    print(gen_nodes)
    print(load_nodes)

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
    print("\nNo. of subnetworks: {}".format(subnetwork_cnt+1))
    for key in subnetwork_dict.keys():
        print("\nsubnetwork : ", key)
        print([node_dict[i] for i in subnetwork_dict[key]])

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
        print(net_gen)
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

    print("\nPATH_REDUNDANCY: ", path_redundancy)
    print("MIN_FEASIBLE_SUBNETWORKS: ", min_feasible_subnetworks)

    return path_redundancy, min_feasible_subnetworks


def remove_duplicate_nodes(nodes):
    node_list = []
    for node in nodes:
        if node not in node_list:
            node_list.append(node)
    return node_list


if __name__ == "__main__":
    ###########
    node_dict = {0: '116', 1: '1', 2: '3', 3: '5', 4: '7', 5: '8', 6: '9', 7: '13', 8: '14', 9: '15', 10: '18',
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
                 118: '114', 119: '121', 120: '61', 121: '123', 122: '122', 123: '1_a', 124: '2_b', 125: '4_c',
                 126: '5_c', 127: '6_c', 128: '7_a', 129: '9_a', 130: '10_a', 131: '11_a', 132: '12_b', 133: '16_c',
                 134: '17_c', 135: '19_a', 136: '20_a', 137: '22_b', 138: '24_c', 139: '28_a', 140: '29_a', 141: '30_c',
                 142: '31_c', 143: '32_c', 144: '33_a', 145: '34_c', 146: '35_a', 147: '37_a', 148: '38_b', 149: '39_b',
                 150: '41_c', 151: '42_a', 152: '43_b', 153: '45_a', 154: '46_a', 155: '47_a', 156: '48_a', 157: '49_a',
                 158: '50_c', 159: '51_a', 160: '52_a', 161: '53_a', 162: '55_a', 163: '56_b', 164: '58_b', 165: '59_b',
                 166: '60_a', 167: '62_c', 168: '63_a', 169: '64_b', 170: '65_a', 171: '66_c', 172: '68_a', 173: '69_a',
                 174: '70_a', 175: '71_a', 176: '73_c', 177: '74_c', 178: '75_c', 179: '76_a', 180: '77_b', 181: '79_a',
                 182: '80_b', 183: '82_a', 184: '83_c', 185: '84_c', 186: '85_c', 187: '86_b', 188: '87_b', 189: '88_a',
                 190: '90_b', 191: '92_c', 192: '94_a', 193: '95_b', 194: '96_b', 195: '98_a', 196: '99_b',
                 197: '100_c', 198: '102_c', 199: '103_c', 200: '104_c', 201: '106_b', 202: '107_b', 203: '109_a',
                 204: '111_a', 205: '112_a', 206: '113_a', 207: '114_a'}

    ###########
    bus_graph = {0: [1], 1: [0, 85, 4, 3], 85: [1], 86: [3], 3: [1, 87, 86], 87: [3], 4: [1, 90], 6: [90, 88],
                     88: [89, 6], 89: [88], 90: [4, 6, 21], 91: [92], 92: [91, 21], 11: [93, 94, 22, 21], 93: [11],
                     94: [11, 95], 95: [94, 17], 17: [20, 18, 95], 18: [17, 19], 19: [18], 20: [97, 96, 17], 96: [20],
                     97: [20], 21: [90, 11, 33, 92], 22: [11, 100, 98], 98: [22, 24], 24: [98, 99], 99: [24],
                     100: [22, 26], 26: [100, 28, 101], 101: [26], 28: [29, 102, 26], 102: [28], 29: [103, 30, 28],
                     103: [29], 30: [29, 31], 31: [30, 32], 32: [31], 33: [21, 34], 34: [33, 38, 36], 36: [34, 82],
                     82: [36], 38: [104, 39, 34], 104: [38], 39: [38, 40, 45], 40: [39, 41], 41: [40, 42], 42: [41, 43],
                     43: [42, 105], 105: [43], 45: [39, 65, 46, 49], 46: [45, 47], 47: [46, 106], 106: [47],
                     49: [45, 50, 51], 50: [49, 107], 107: [50], 51: [49, 52, 58], 52: [51, 108], 108: [52, 54],
                     54: [108, 57], 56: [57, 83], 83: [56], 57: [54, 109, 56], 109: [57], 58: [59, 51],
                     59: [111, 58, 110], 110: [59], 111: [112, 59], 112: [113, 111], 113: [63, 112], 63: [114, 113],
                     114: [63], 65: [70, 66, 45], 66: [65, 67], 67: [66], 70: [73, 71, 65], 71: [70, 115], 115: [71],
                     73: [75, 116, 70], 116: [73], 75: [117, 73], 117: [75], 77: [78], 78: [77, 118],
                     118: [78]}

    ###########
    active_nodes = ['1_a', '2_b', '4_c', '5_c', '6_c', '7_a', '9_a', '10_a', '11_a', '12_b', '16_c', '17_c', '19_a', '20_a',
              '22_b', '24_c', '28_a', '29_a', '30_c', '31_c', '32_c', '33_a', '34_c', '35_a', '37_a', '38_b', '39_b',
              '41_c', '42_a', '43_b', '45_a', '46_a', '47_a', '48_a', '49_a', '50_c', '51_a', '52_a', '53_a', '55_a',
              '56_b', '58_b', '59_b', '60_a', '62_c', '63_a', '64_b', '65_a', '66_c', '68_a', '69_a', '70_a', '71_a',
              '73_c', '74_c', '75_c', '76_a', '77_b', '79_a', '80_b', '82_a', '83_c', '84_c', '85_c', '86_b', '87_b',
              '88_a', '90_b', '92_c', '94_a', '95_b', '96_b', '98_a', '99_b', '100_c', '102_c', '103_c', '104_c',
              '106_b', '107_b', '109_a', '111_a', '112_a', '113_a', '114_a']  # gens+loads

    ###########
    gens = ['28_a', '35_a', '47_a', '60_a', '76_a', '98_a', '109_a', '34_c', '53_a', '64_b', '51_a', '80_b']  # only gens

    gen_nodes = []
    for __ind in np.arange(len(active_nodes)):
        if active_nodes[__ind] in gens:
            gen_nodes.append(__ind)
    gen_nodes = remove_duplicate_nodes(gen_nodes)

    gen_node_cap = np.zeros(len(active_nodes))
    for gen__ind in np.arange(len(active_nodes)):
        if gen__ind in gen_nodes:
            gen_node_cap[gen__ind] = 100

    ###########
    diesel_left = np.zeros((len(active_nodes), 1))
    for gen__ind in np.arange(len(active_nodes)):
        if gen__ind in gen_nodes:
            diesel_left[gen__ind, 0] = 100

    ###########
    ess_left = 20 * np.ones(len(active_nodes))  # in kWh

    ###########
    p_cr = 5 * np.random.rand(len(active_nodes), 2)

    ###########
    coeff_b = 0.05

    ###########
    time_step = 0.25

    ###########
    p_pv = 50 * np.random.rand(len(active_nodes), 2)

    ###########
    bat_s_cap = 40 * np.ones((len(active_nodes), 1))

    ###########
    bat_p_cap = 20 * np.ones((len(active_nodes), 1))

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