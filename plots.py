from email.errors import InvalidMultipartContentTransferEncodingDefect
import matplotlib.pyplot as plt
import numpy as np

reqomp_timings_types = ['constr', 'circ-to-dag', 'dag-to-dep', 'uncomp', 'dep-to-dag', 'dag-to-circ', 'decomp']
qiskit_timings_types = ['constr', 'decomp']
unqomp_timings_types = ['constr', 'uncomp-all', 'decomp']


def get_vals_from_last_line(filename, pos_bef_last):
    file = open(filename)
    print('opend ', filename)
    last_lines = ["" for i in range(pos_bef_last + 1)]
    i = 0
    for line in file: #we only want last lne with nb qbs and gates
        last_lines[i] = line
        i = (i + 1) % (pos_bef_last + 1)
    last_line = last_lines[i]
    assert last_line.find(';') != -1
    vals = last_line.split(";")
    if len(vals) == 5:
        print("for " + filename + " vals " + str(vals))
        vals_as_dict = {'a': int(vals[0]), 'q': int(vals[1]), 'g': int(vals[2]) + int(vals[3]), 'cx':int(vals[3]), 'd' : int(vals[4])}
    else:
        vals_as_dict = {'q': int(vals[0]), 'g': int(vals[1]) + int(vals[2]), 'cx':int(vals[2]), 'd' : int(vals[3])}
    print("for " + filename + " vals as dict" + str(vals_as_dict))
    # a = nb ancillas
    # q = nb qbs
    # g = nb gates tot
    # cx = nb cx gates
    # d = depth
    file.close()
    return vals_as_dict

def get_timings_reqomp(filename):
    # time construction circuit
    # time conversion to dag
    # time conversion to dep graph
    # time uncomputation
    # time conversion to dag
    # time conversion to circuit
    # time decomposition
    # nb qbs; nb gates tot; nb cx for reqomp
    file = open(filename)
    print('opened ', filename)
    vals = {}
    i = 0
    for line in file:
        if i >= len(reqomp_timings_types):
            break
        print(line)
        vals[reqomp_timings_types[i]] = float(line)
        i += 1
    print(vals)
    file.close()
    return vals

def get_timings_qiskit(filename):
    #time construction with unqomp
    # time decomposition
    # nb qbs; nb gates tot; nb cx for qiskit
    file = open(filename)
    print('opened ', filename)
    vals = {}
    i = 0
    for line in file:
        if i >= len(qiskit_timings_types):
            break
        print(line)
        vals[qiskit_timings_types[i]] = float(line)
        i += 1
    print(vals)
    file.close()
    return vals

def get_timings_unqomp(filename):
    #time construction circuit
    #time unqomp tout compris
    # time decomposition
    # nb qbs; nb gates tot; nb cx for reqomp
    file = open(filename)
    print('opened ', filename)
    vals = {}
    i = 0
    for line in file:
        if i >= len(unqomp_timings_types):
            break
        print(line)
        vals[unqomp_timings_types[i]] = float(line)
        i += 1
    print(vals)
    file.close()
    return vals

def reads_up_perf_no_params(file_prefix, n_min, n_max, step, anc_max_function, pos_bef_last):
    perf_values = {}
    for n in range(n_min, n_max, step):
        for n_anc in range(1, anc_max_function(n) + 1):
            full_name = "evaluation_results/" + file_prefix + "_" + str(n) + "_" + str(n_anc)
            print(full_name)
            try:
                perf_values[(n, n_anc)] = get_vals_from_last_line(full_name, pos_bef_last)
            except Exception:
                pass
        full_name = "evaluation_results/" + file_prefix + "_" + str(n) + "_u"
        #try:
        if True:
            perf_values[(n, 'u')] = get_vals_from_last_line(full_name, pos_bef_last)
        #except Exception:
        #    pass
        full_name = "evaluation_results/" + file_prefix + "_" + str(n) + "_q"
        try:
            perf_values[(n, 'q')] = get_vals_from_last_line(full_name, pos_bef_last)
        except Exception:
            pass
    # perf_values: dict wiht key (size_example, nb_qbs_for_reqomp) or (size_example, 'u') for Unqomp or (size_example, 'q') for qiskit, and value is a dict with keys 'q' for nb qubits, 'g' for nb gates(all) and 'cx' for nb cx gates
    return perf_values

def reads_up_timings_no_params(file_prefix, n_min, n_max, step, anc_max_function):
    timing_values = {}
    for n in range(n_min, n_max, step):
        for n_anc in range(1, anc_max_function(n) + 1):
            full_name = "evaluation_results/" + file_prefix + "_" + str(n) + "_" + str(n_anc)
            print(full_name)
            try:
                timing_values[(n, n_anc)] = get_timings_reqomp(full_name)
            except Exception:
                pass
        full_name = "evaluation_results/" + file_prefix + "_" + str(n) + "_u"
        try:
            timing_values[(n, 'u')] = get_timings_unqomp(full_name)
        except Exception:
            pass
        full_name = "evaluation_results/" + file_prefix + "_" + str(n) + "_q"
        try:
            timing_values[(n, 'q')] = get_timings_qiskit(full_name)
        except Exception:
            pass
    # dict with keys: (size, 'u'), (size, 'q') and (size, nb_anc) for reqomp
    # and dict[k]: dict with keys:
    # reqomp_timings_types = ['constr', 'circ-to-dag', 'dag-to-dep', 'uncomp', 'dep-to-dag', 'dag-to-circ', 'decomp']
    # qiskit_timings_types = ['constr', 'decomp']
    # unqomp_timings_types = ['constr', 'uncomp-all', 'decomp'] 
    return timing_values


def reads_up_perf_no_anc_nbs(file_prefix, n_min, n_max, step, pos_bef_last):
    perf_values = {}
    for n in range(n_min, n_max, step):
        full_name = "evaluation_results/" + file_prefix + "_" + str(n)
        print(full_name)
        try:
            perf_values[(n, 0)] = get_vals_from_last_line(full_name, pos_bef_last)
        except Exception:
            pass
        full_name = "evaluation_results/" + file_prefix + "_" + str(n) + "_u"
        try:
            perf_values[(n, 'u')] = get_vals_from_last_line(full_name, pos_bef_last)
        except Exception:
            pass
        full_name = "evaluation_results/" + file_prefix + "_" + str(n) + "_q"
        try:
            perf_values[(n, 'q')] = get_vals_from_last_line(full_name, pos_bef_last)
        except Exception:
            pass

    # perf_values: dict wiht key (size_example, nb_qbs_for_reqomp) or (size_example, 'u') for Unqomp or (size_example, 'q') for qiskit, and value is a dict with keys 'q' for nb qubits, 'g' for nb gates(all) and 'cx' for nb cx gates
    return perf_values

def reads_up_timings_no_anc_nbs(file_prefix, n_min, n_max, step):
    perf_values = {}
    for n in range(n_min, n_max, step):
        full_name = "evaluation_results/" + file_prefix + "_" + str(n)
        print(full_name)
        try:
            perf_values[(n, 0)] = get_timings_reqomp(full_name)
        except Exception:
            pass
        full_name = "evaluation_results/" + file_prefix + "_" + str(n) + "_u"
        try:
            perf_values[(n, 'u')] = get_timings_unqomp(full_name)
        except Exception:
            pass
        full_name = "evaluation_results/" + file_prefix + "_" + str(n) + "_q"
        try:
            perf_values[(n, 'q')] = get_timings_qiskit(full_name)
        except Exception:
            pass

    return perf_values

def makes_plot_timings_cleaner(file_prefix, n_min, n_max, step, anc_max_function, size_n, is_no_param):
    if is_no_param:
        timing_values = reads_up_timings_no_params(file_prefix, n_min, n_max, step, anc_max_function)
    else:
        timing_values = reads_up_timings_no_anc_nbs(file_prefix, n_min, n_max, step)
    fig, axesL = plt.subplots(ncols = 1, figsize=(36, 12))         # creates a nice, Puschel-style plot
    axes = axesL

    qbs_reqomp = []
    vals_reqomp = []
    decomp_reqomp = []
    for (cur_size, cur_nb_qbs) in timing_values:
        print(cur_size, cur_nb_qbs)
        if cur_size == size_n and isinstance(cur_nb_qbs, int):
            qbs_reqomp.append(cur_nb_qbs)
            t = timing_values[(cur_size, cur_nb_qbs)]
            val_t = t['circ-to-dag'] + t['dag-to-dep'] + t['dag-to-dep'] + t['dep-to-dag'] + t['dag-to-circ']
            vals_reqomp.append(val_t)
            decomp_reqomp.append(t['decomp'])
    print(qbs_reqomp)
    axes.plot(qbs_reqomp, vals_reqomp, 'bo', label = 'Reqomp')
    axes.plot(qbs_reqomp, decomp_reqomp, 'ys', label = 'Decomp Reqomp')
    axes.plot([max(qbs_reqomp)], [timing_values[(size_n, 'u')]['uncomp-all']], 'gx', label = 'Unqomp')
    #axes.plot([max(qbs_reqomp)], [timing_values[(size_n, 'q')]['constr'] + timing_values[(size_n, 'q')]['decomp']], 'r+', label = 'Qiskit')
    fig.suptitle("Timings for " + file_prefix + " size" + str(size_n))
    axes.legend()
                
    plt.savefig(file_prefix + str(size_n) + '_timingsclean.png')
    plt.close()

def identity(n):
    return n
def min1(n):
    return n-1
def min2(n):
    return n-2
def makes_trade_off_plot(file_prefix, n_min, n_max, step, anc_max_function, pos_bef_last, is_no_anc_nbs, gates_index = 'g'):
    if is_no_anc_nbs:
        perf_values = reads_up_perf_no_anc_nbs(file_prefix, n_min, n_max, step, pos_bef_last)
    else:
        perf_values = reads_up_perf_no_params(file_prefix, n_min, n_max, step, anc_max_function, pos_bef_last)
    for n_size in range(n_min, n_max, step):
        fig, axesL = plt.subplots(ncols = 1, figsize=(36, 12))         # creates a nice, Püschel-style plot
        axes = axesL
        qbs_reqomp = []
        vals_reqomp = []
        for (cur_size, cur_nb_qbs) in perf_values:
            if cur_size == n_size and isinstance(cur_nb_qbs, int):
                qbs_reqomp.append(perf_values[(cur_size, cur_nb_qbs)]['q'])
                vals_reqomp.append(perf_values[(cur_size, cur_nb_qbs)][gates_index])
        axes.plot(qbs_reqomp, vals_reqomp, color='b', label = 'Reqomp')
        print(vals_reqomp)
        axes.plot([perf_values[(n_size, 'u')]['q']], [perf_values[(n_size, 'u')][gates_index]], 'gx', label = 'Unqomp')
        axes.plot([perf_values[(n_size, 'q')]['q']], [perf_values[(n_size, 'q')][gates_index]], 'r+', label = 'Qiskit')
        fig.suptitle("Trade offs for " + file_prefix + " size" + str(n_size))
                
        plt.savefig(file_prefix + str(n_size) + gates_index + '_trade_offs.png')
        plt.close()


def proper_name(file_name):
        if file_name == 'adder':
            return 'Adder'
        if file_name == 'grover':
            return 'Grover'
        if file_name == 'dj':
            return 'Deutsch-Jozsa'
        if file_name == 'mcx':
            return 'MCX'
        if file_name == 'mcry':
            return 'MCRY'
        if file_name == 'polypr':
            return 'PolynomialPauliR'
        if file_name == 'intcomp':
            return 'IntegerComparator'
        if file_name == 'plr':
            return 'PiecewiseLinearR'
        if file_name == 'wa':
            return 'WeightedAdder'
        if file_name == 'mult':
            return 'Multiplier'
        print("AAAAAAAH " + file_name)

def makes_trade_off_plot_given_fig(axes, n_size, file_prefix, n_min, n_max, step, anc_max_function, pos_bef_last, is_no_anc_nbs, gates_index, markers = False):
    if is_no_anc_nbs:
        perf_values = reads_up_perf_no_anc_nbs(file_prefix, n_min, n_max, step, pos_bef_last)
    else:
        perf_values = reads_up_perf_no_params(file_prefix, n_min, n_max, step, anc_max_function, pos_bef_last)

    qbs_reqomp = []
    vals_reqomp = []
    nb_qbs_non_anc = -1
    for (cur_size, cur_nb_qbs) in perf_values:
        if cur_size == n_size and isinstance(cur_nb_qbs, int):
            qbs_reqomp.append(perf_values[(cur_size, cur_nb_qbs)]['a'])
            vals_reqomp.append(perf_values[(cur_size, cur_nb_qbs)][gates_index])
            new_n_non_anc = perf_values[(cur_size, cur_nb_qbs)]['q'] - perf_values[(cur_size, cur_nb_qbs)]['a']
            print("n non anc " + str(new_n_non_anc) + " for " + str((cur_size, cur_nb_qbs))) 
            print(perf_values[(cur_size, cur_nb_qbs)])
            if nb_qbs_non_anc != -1 and new_n_non_anc != nb_qbs_non_anc:
                assert False
            nb_qbs_non_anc = new_n_non_anc
    print("for " + file_prefix + " reqoimp " + str(qbs_reqomp) + " " + str(vals_reqomp))
    if markers:
        axes.plot(qbs_reqomp, vals_reqomp, marker = 'o', color = '#00701a', label = 'Reqomp')
    else:
        axes.plot(qbs_reqomp, vals_reqomp, color = '#00701a', label = 'Reqomp')
    axes.plot([perf_values[(n_size, 'u')]['q']  - nb_qbs_non_anc], [perf_values[(n_size, 'u')][gates_index]], marker = 'P', color = '#ff7961', label = 'Unqomp')
    if False:
        axes.plot([perf_values[(n_size, 'q')]['q'] - nb_qbs_non_anc], [perf_values[(n_size, 'q')][gates_index]], marker = 'v', color = '#ba000d', label = 'Qiskit')
    axes.set_ylim(bottom=0)
    axes.set_xlim(left=0)
    lab_y = "\# gates" if gates_index == 'g' else "circuit depth"
    axes.set(xlabel = '\# ancillae', ylabel = lab_y)
    axes.set_title(proper_name(file_prefix))

def makes_trade_offs_plots_one_file():
    fig, axesL = plt.subplots(nrows = 2, ncols = 2, figsize=(24, 24))
    gates_index = 'g'
    makes_trade_off_plot_given_fig(axesL[0][0], 200, 'mcx', 200, 201, 1, min2, 0, False, gates_index)
    makes_trade_off_plot_given_fig(axesL[0][1], 40, 'plr', 40, 41, 3, identity, 4, False, gates_index)
    makes_trade_off_plot_given_fig(axesL[1][0], 100, 'intcomp', 100, 101, 1, min1, 1, False, gates_index)
    makes_trade_off_plot_given_fig(axesL[1][1], 100, 'adder', 100, 101, 1, identity, 0, False, gates_index)
    fig.tight_layout()
    plt.savefig('gate_counts_big.png')

    fig, axesL = plt.subplots(nrows = 2, ncols = 2, figsize=(24, 24))
    gates_index = 'd'
    makes_trade_off_plot_given_fig(axesL[0][0], 200, 'mcx', 200, 201, 1, min2, 0, False, gates_index)
    makes_trade_off_plot_given_fig(axesL[0][1], 40, 'plr', 40, 41, 3, identity, 4, False, gates_index)
    makes_trade_off_plot_given_fig(axesL[1][0], 100, 'intcomp', 100, 101, 1, min1, 1, False, gates_index)
    makes_trade_off_plot_given_fig(axesL[1][1], 100, 'adder', 100, 101, 1, identity, 0, False, gates_index)
    fig.tight_layout()
    plt.savefig('depth_big.png')

    fig, axesL = plt.subplots(nrows = 2, ncols = 2, figsize=(24, 24))
    gates_index = 'g'
    makes_trade_off_plot_given_fig(axesL[0][0], 12, 'mcx', 12, 13, 1, min2, 0, False, gates_index, True)
    makes_trade_off_plot_given_fig(axesL[0][1], 6, 'plr', 6, 7, 3, identity, 4, False, gates_index, True)
    makes_trade_off_plot_given_fig(axesL[1][0], 12, 'intcomp', 12, 13, 1, min1, 1, False, gates_index, True)
    makes_trade_off_plot_given_fig(axesL[1][1], 12, 'adder', 12, 13, 1, identity, 0, False, gates_index, True)
    fig.tight_layout()
    plt.savefig('gate_counts_small.png')

    fig, axesL = plt.subplots(nrows = 2, ncols = 2, figsize=(24, 24))
    gates_index = 'd'
    makes_trade_off_plot_given_fig(axesL[0][0], 12, 'mcx', 12, 13, 1, min2, 0, False, gates_index, True)
    makes_trade_off_plot_given_fig(axesL[0][1], 6, 'plr', 6, 7, 3, identity, 4, False, gates_index, True)
    makes_trade_off_plot_given_fig(axesL[1][0], 12, 'intcomp', 12, 13, 1, min1, 1, False, gates_index, True)
    makes_trade_off_plot_given_fig(axesL[1][1], 12, 'adder', 12, 13, 1, identity, 0, False, gates_index, True)
    fig.tight_layout()
    plt.savefig('depth_small.png')

def gets_vals_one_example(file_prefix, n_min, n_max, step, anc_max_function, pos_bef_last, is_no_anc_nbs, n_chosen, f):
    if is_no_anc_nbs:
        perf_values = reads_up_perf_no_anc_nbs(file_prefix, n_min, n_max, step, pos_bef_last)
    else:
        perf_values = reads_up_perf_no_params(file_prefix, n_min, n_max, step, anc_max_function, pos_bef_last)
    qbs_reqomp = []
    anc_reqomp = []
    vals_reqomp_cx = []
    vals_reqomp_gates = []
    vals_reqomp_depth = []
    for (cur_size, cur_nb_qbs) in perf_values:
        if cur_size == n_chosen and isinstance(cur_nb_qbs, int):
            qbs_reqomp.append(perf_values[(cur_size, cur_nb_qbs)]['q'])
            anc_reqomp.append(perf_values[(cur_size, cur_nb_qbs)]['a'])
            vals_reqomp_cx.append(perf_values[(cur_size, cur_nb_qbs)]['cx'])
            vals_reqomp_gates.append(perf_values[(cur_size, cur_nb_qbs)]['g'])
            vals_reqomp_depth.append(perf_values[(cur_size, cur_nb_qbs)]['d'])

    # remove useless ancillae: when adding more ancillae does not change operations, it means they are not used.
    #find smallest i such that same nb gates
    v_eq = [i for i in range(len(qbs_reqomp)) if vals_reqomp_gates[i] == vals_reqomp_gates[-1]]
    last_index = min(v_eq)
    qbs_reqomp = qbs_reqomp[:last_index + 1]
    anc_reqomp = anc_reqomp[:last_index + 1]
    vals_reqomp_cx = vals_reqomp_cx[:last_index + 1]
    vals_reqomp_gates = vals_reqomp_gates[:last_index + 1]
    vals_reqomp_depth = vals_reqomp_depth[:last_index + 1]

    print(perf_values)
    minVol = -1
    n_qbs_min = []
    nb_gates_for_same_qb_u = -1
    nb_cx_gates_for_same_qb_u = -1
    print(qbs_reqomp)
    for i in range(len(qbs_reqomp)):
        vol = qbs_reqomp[i] * vals_reqomp_depth[i]
        print("for " + file_prefix + " and nb qbs " + str(qbs_reqomp[i]) + " we have vol " + str(vol))
        if vol == minVol:
            n_qbs_min.append(qbs_reqomp[i])
        if minVol == -1 or vol < minVol:
            minVol = vol
            n_qbs_min = [qbs_reqomp[i]]
        if qbs_reqomp[i] == perf_values[(n_chosen, 'u')]['q']:
            nb_gates_for_same_qb_u = vals_reqomp_gates[i]
            nb_cx_gates_for_same_qb_u = vals_reqomp_cx[i]
            depth_for_same_qb_u = vals_reqomp_depth[i]


    volUnqomp = perf_values[(n_chosen, 'u')]['q'] * perf_values[(n_chosen, 'u')]['d']
    def reduction(list_vals):
        v_max = max(list_vals)
        v_min = min(list_vals)
        return format((v_max - v_min) / v_max * 100, '.1f')

    unqomp_qbs = perf_values[(n_chosen, 'u')]['q']
    unqomp_cx = perf_values[(n_chosen, 'u')]['cx']
    unqomp_gates = perf_values[(n_chosen, 'u')]['g']
    nb_anc_unqomp = unqomp_qbs - (qbs_reqomp[0] - anc_reqomp[0])
    print(proper_name(file_prefix))
    print("unqomp vals " + str(perf_values[(n_chosen, 'u')]))
    print("reqomp qbs " + str(qbs_reqomp))
    print("reqomp anc " + str(anc_reqomp))
    print(vals_reqomp_gates)

    def reduction_vs_unqomp(reqomp_val, unqomp_val):
        return format((reqomp_val - unqomp_val) / unqomp_val * 100, '.1f')

    def find_qubit_red(qubit_red_obj): #the max nb qbs tq red vs unqomp is >= red_obj
        cur_best_anc = -1
        cur_best_cx = -1
        cur_best_gate = -1
        for i in range(len(qbs_reqomp)):
            if (nb_anc_unqomp - anc_reqomp[i])/ nb_anc_unqomp >= qubit_red_obj:
                if anc_reqomp[i] > cur_best_anc:
                    cur_best_anc = anc_reqomp[i]
                    cur_best_cx = vals_reqomp_cx[i]
                    cur_best_gate = vals_reqomp_gates[i]
        if cur_best_anc == -1:
            return "x, x, x"
        else:
            return reduction_vs_unqomp(cur_best_anc, nb_anc_unqomp) + ", " + reduction_vs_unqomp(cur_best_cx, unqomp_cx) + ", " + reduction_vs_unqomp(cur_best_gate, unqomp_gates)

    def find_red_max():
        cur_best_anc = max(anc_reqomp)
        for i in range(len(qbs_reqomp)):
            if cur_best_anc == anc_reqomp[i]:
                cur_best_anc = anc_reqomp[i]
                cur_best_cx = vals_reqomp_cx[i]
                cur_best_gate = vals_reqomp_gates[i]
                return reduction_vs_unqomp(cur_best_anc, nb_anc_unqomp) + ", " + reduction_vs_unqomp(cur_best_cx, unqomp_cx) + ", " + reduction_vs_unqomp(cur_best_gate, unqomp_gates)
    
    def find_red_min():
        cur_best_anc = min(anc_reqomp)
        for i in range(len(qbs_reqomp)):
            if cur_best_anc == anc_reqomp[i]:
                cur_best_anc = anc_reqomp[i]
                cur_best_cx = vals_reqomp_cx[i]
                cur_best_gate = vals_reqomp_gates[i]
                return reduction_vs_unqomp(cur_best_anc, nb_anc_unqomp) + ", " + reduction_vs_unqomp(cur_best_cx, unqomp_cx) + ", " + reduction_vs_unqomp(cur_best_gate, unqomp_gates)

    if False:
        output_line = proper_name(file_prefix) + ", " + str(min(qbs_reqomp)) + ", " + str(max(qbs_reqomp)) + ", " + str(reduction(qbs_reqomp)) + "\\\\, "
        output_line += str(min(vals_reqomp_cx)) + ", " + str(max(vals_reqomp_cx)) + "," + str(reduction(vals_reqomp_cx)) + ", "
        output_line += str(min(vals_reqomp_gates)) + ", " + str(max(vals_reqomp_gates)) + ","+ str(reduction(vals_reqomp_gates)) + ", "
        output_line += str(min(vals_reqomp_depth)) + ", " + str(max(vals_reqomp_depth)) + ","+ str(reduction(vals_reqomp_depth)) + ", "
        output_line +=  str(minVol) + ", (" + str(n_qbs_min[0]) + "), "
        output_line += str(perf_values[(n_chosen, 'u')]['q']) + ", " + str(perf_values[(n_chosen, 'u')]['cx']) + ", " + str(perf_values[(n_chosen, 'u')]['g']) + ", " + str(perf_values[(n_chosen, 'u')]['d']) + ", "
        output_line += str(volUnqomp) + ","
        output_line += format((volUnqomp - minVol) / volUnqomp * 100, '.1f') + " \\\\" +  ","
        output_line += format((perf_values[(n_chosen, 'u')]['g'] - nb_gates_for_same_qb_u) / perf_values[(n_chosen, 'u')]['g'] * 100, '.1f') + ","
        output_line += format((perf_values[(n_chosen, 'u')]['cx'] - nb_cx_gates_for_same_qb_u) / perf_values[(n_chosen, 'u')]['cx'] * 100, '.1f') + ","
        output_line += format((perf_values[(n_chosen, 'u')]['d'] - depth_for_same_qb_u) / perf_values[(n_chosen, 'u')]['d'] * 100, '.1f') + " \\\\"
        
    output_line = proper_name(file_prefix) + ", " + find_red_min() + ", " + find_qubit_red(0.75) + ", " + find_qubit_red(0.5) + ", " + find_qubit_red(0.25) + ", " + find_red_max() + "\\\\"

    print(output_line)
    print(output_line, file = f)

def gets_vals_one_example_app(file_prefix, n_min, n_max, step, anc_max_function, pos_bef_last, is_no_anc_nbs, n_chosen, f, is_2nd = False):
    if is_no_anc_nbs:
        perf_values = reads_up_perf_no_anc_nbs(file_prefix, n_min, n_max, step, pos_bef_last)
    else:
        perf_values = reads_up_perf_no_params(file_prefix, n_min, n_max, step, anc_max_function, pos_bef_last)
    qbs_reqomp = []
    anc_reqomp = []
    vals_reqomp_cx = []
    vals_reqomp_gates = []
    vals_reqomp_depth = []
    n_qbs_correspondant = {}
    for (cur_size, cur_nb_qbs) in perf_values:
        if cur_size == n_chosen and isinstance(cur_nb_qbs, int):
            qbs_reqomp.append(perf_values[(cur_size, cur_nb_qbs)]['q'])
            n_qbs_correspondant[perf_values[(cur_size, cur_nb_qbs)]['q']] = cur_nb_qbs
            anc_reqomp.append(perf_values[(cur_size, cur_nb_qbs)]['a'])
            vals_reqomp_cx.append(perf_values[(cur_size, cur_nb_qbs)]['cx'])
            vals_reqomp_gates.append(perf_values[(cur_size, cur_nb_qbs)]['g'])
            vals_reqomp_depth.append(perf_values[(cur_size, cur_nb_qbs)]['d'])

    # remove useless ancillae: when adding more ancillae does not change operations, it means they are not used.
    #find smallest i such that same nb gates
    v_eq = [i for i in range(len(qbs_reqomp)) if vals_reqomp_gates[i] == vals_reqomp_gates[-1]]
    last_index = min(v_eq)
    qbs_reqomp = qbs_reqomp[:last_index + 1]
    anc_reqomp = anc_reqomp[:last_index + 1]
    vals_reqomp_cx = vals_reqomp_cx[:last_index + 1]
    vals_reqomp_gates = vals_reqomp_gates[:last_index + 1]
    vals_reqomp_depth = vals_reqomp_depth[:last_index + 1]

    print(perf_values)

    unqomp_qbs = perf_values[(n_chosen, 'u')]['q']
    unqomp_cx = perf_values[(n_chosen, 'u')]['cx']
    unqomp_gates = perf_values[(n_chosen, 'u')]['g']
    nb_anc_unqomp = unqomp_qbs - (qbs_reqomp[0] - anc_reqomp[0])
    perf_values[(n_chosen, 'u')]['a'] = nb_anc_unqomp
    print(proper_name(file_prefix))
    print("unqomp vals " + str(perf_values[(n_chosen, 'u')]))
    print("reqomp qbs " + str(qbs_reqomp))
    print("reqomp anc " + str(anc_reqomp))
    print(vals_reqomp_gates)

    def print_value(dict_of_vals):
        return str(dict_of_vals['q']) + ", " + str(dict_of_vals['a']) + "," + str(dict_of_vals['cx']) + "," + str(dict_of_vals['g']) + "," + str(dict_of_vals['d'])

    def find_qubit_red(qubit_red_obj): #the max nb qbs tq red vs unqomp is >= red_obj
        cur_best_anc = -1
        cur_best_cx = -1
        cur_best_gate = -1
        pos_found = -1
        for i in range(len(qbs_reqomp)):
            if (nb_anc_unqomp - anc_reqomp[i])/ nb_anc_unqomp >= qubit_red_obj:
                if anc_reqomp[i] > cur_best_anc:
                    pos_found = i
                    cur_best_anc = anc_reqomp[i]
                    cur_best_cx = vals_reqomp_cx[i]
                    cur_best_gate = vals_reqomp_gates[i]
        if cur_best_anc == -1:
            return "x, x, x, x, x"
        else:
            return print_value(perf_values[(cur_size, n_qbs_correspondant[qbs_reqomp[pos_found]])])

    def find_red_max():
        cur_best_anc = max(anc_reqomp)
        for i in range(len(qbs_reqomp)):
            if cur_best_anc == anc_reqomp[i]:
                cur_best_anc = anc_reqomp[i]
                cur_best_cx = vals_reqomp_cx[i]
                cur_best_gate = vals_reqomp_gates[i]
                return print_value(perf_values[(cur_size, n_qbs_correspondant[qbs_reqomp[i]])])
    
    def find_red_min():
        cur_best_anc = min(anc_reqomp)
        for i in range(len(qbs_reqomp)):
            if cur_best_anc == anc_reqomp[i]:
                cur_best_anc = anc_reqomp[i]
                cur_best_cx = vals_reqomp_cx[i]
                cur_best_gate = vals_reqomp_gates[i]
                return print_value(perf_values[(cur_size, n_qbs_correspondant[qbs_reqomp[i]])])

    if not is_2nd:
        output_line = proper_name(file_prefix) + ", " + find_red_min() + ", " + find_qubit_red(0.75) + ", " + find_qubit_red(0.5) + "\\\\"
    else:
        output_line = proper_name(file_prefix) + ", " + find_qubit_red(0.25) + ", " + find_red_max() + ", " + print_value(perf_values[(cur_size, 'u')]) + "\\\\"

    print(output_line)
    print(output_line, file = f)

def gets_vals_all_examples():
    def one_print(txt):
        return "qbs red for " + txt + ", cx red for " + txt + ", gates red for " + txt 
    def one_print_col_name(txt):
        return "qbs_red_for_" + txt + ", " + "cx_red_for_" + txt + ", gates_red_for_" + txt
        
    with open("tradeoffssmall.csv", 'w') as f:
        print("Example, " + one_print("min qbs") + ", " + one_print("0.75 red") + ", " + one_print("0.5 red") + ", " + one_print("0.25 red") + ", " + one_print("max qbs"))
        print("example, " + one_print_col_name("min_qbs") + ", " + one_print_col_name("075_qbs") + ", " + one_print_col_name("05_qbs") + ", " + one_print_col_name("025_qbs")+ ", " + one_print_col_name("max_qbs"), file = f)
        gets_vals_one_example('adder', 12, 13, 1, identity, 0, False, 12, f)
        gets_vals_one_example('dj', 10, 11, 1, identity, 0, False, 10, f)
        gets_vals_one_example('grover', 5, 6, 2, identity, 0, False, 5, f)
        gets_vals_one_example('intcomp', 12, 13, 1, identity, 1, False, 12, f)
        gets_vals_one_example('mcry', 12, 13, 1, identity, 0, False, 12, f)
        gets_vals_one_example('mcx', 12, 13, 1, identity, 0, False, 12, f)
        gets_vals_one_example('mult', 5, 6, 1, identity, 0, True, 5, f)
        gets_vals_one_example('plr', 6, 7, 3, identity, 4, False, 6, f)
        gets_vals_one_example('polypr', 5, 6, 1, identity, 0, False, 5, f)
        gets_vals_one_example('wa', 10, 11, 1, identity, 1, True, 10, f)

    with open("tradeoffsbig.csv", 'w') as f:
        print("Example, " + one_print("min qbs") + ", " + one_print("0.75 red") + ", " + one_print("0.5 red") + ", " + one_print("0.25 red") + ", " + one_print("max qbs"))
        print("example, " + one_print_col_name("min_qbs") + ", " + one_print_col_name("075_qbs") + ", " + one_print_col_name("05_qbs") + ", " + one_print_col_name("025_qbs")+ ", " + one_print_col_name("max_qbs"), file = f)
        gets_vals_one_example('adder', 100, 101, 1, identity, 0, False, 100, f)
        gets_vals_one_example('dj', 100, 101, 1, identity, 0, False, 100, f)
        gets_vals_one_example('grover', 10, 11, 2, identity, 0, False, 10, f)
        gets_vals_one_example('intcomp', 100, 101, 1, identity, 1, False, 100, f)
        gets_vals_one_example('mcry', 200, 201, 1, identity, 0, False, 200, f)
        gets_vals_one_example('mcx', 200, 201, 1, identity, 0, False, 200, f)
        gets_vals_one_example('mult', 16, 17, 1, identity, 0, True, 16, f)
        gets_vals_one_example('plr', 40, 41, 3, identity, 4, False, 40, f)
        gets_vals_one_example('polypr', 10, 11, 1, identity, 0, False, 10, f)
        gets_vals_one_example('wa', 20, 21, 1, identity, 1, True, 20, f)

def gets_vals_all_examples_appendix():
    def one_print(txt):
        return "qbs for " + txt + ", anc for " + txt + ", cx gates for " + txt + ", gates for " + txt + ", depth for " + txt 
    def one_print_col_name(txt):
        return "qbs_for " + txt + ", anc_for " + txt + ", cx_gates_for " + txt + ", gates_for " + txt + ", depth_for " + txt 
        
    for is_snd in [False, True]:
        name = "tradeoffssmall_app.csv" if not is_snd else "tradeoffssmall_app2.csv"
        with open(name, 'w') as f:
            print("Example, " + one_print("min qbs") + ", " + one_print("0.75 red") + ", " + one_print("0.5 red") + ", " + one_print("0.25 red") + ", " + one_print("max qbs") + ", " + one_print("unqomp"))
            if not is_snd:
                print("example, " + one_print_col_name("min_qbs") + ", " + one_print_col_name("075_qbs") + ", " + one_print_col_name("05_qbs"), file = f)# + ", " + one_print_col_name("025_qbs")+ ", " + one_print_col_name("max_qbs") + ", " + one_print_col_name("unqomp"), file = f)
            else:
                print("example, " + one_print_col_name("025_qbs")+ ", " + one_print_col_name("max_qbs") + ", " + one_print_col_name("unqomp"), file = f)
            
            gets_vals_one_example_app('adder', 12, 13, 1, identity, 0, False, 12, f, is_snd)
            gets_vals_one_example_app('dj', 10, 11, 1, identity, 0, False, 10, f, is_snd)
            gets_vals_one_example_app('grover', 5, 6, 2, identity, 0, False, 5, f, is_snd)
            gets_vals_one_example_app('intcomp', 12, 13, 1, identity, 1, False, 12, f, is_snd)
            gets_vals_one_example_app('mcry', 12, 13, 1, identity, 0, False, 12, f, is_snd)
            gets_vals_one_example_app('mcx', 12, 13, 1, identity, 0, False, 12, f, is_snd)
            gets_vals_one_example_app('mult', 5, 6, 1, identity, 0, True, 5, f, is_snd)
            gets_vals_one_example_app('plr', 6, 7, 3, identity, 4, False, 6, f, is_snd)
            gets_vals_one_example_app('polypr', 5, 6, 1, identity, 0, False, 5, f, is_snd)
            gets_vals_one_example_app('wa', 10, 11, 1, identity, 1, True, 10, f, is_snd)

    for is_snd in [False, True]:
        name = "tradeoffsbig_app.csv" if not is_snd else "tradeoffsbig_app2.csv"
        with open(name, 'w') as f:
            print("Example, " + one_print("min qbs") + ", " + one_print("0.75 red") + ", " + one_print("0.5 red") + ", " + one_print("0.25 red") + ", " + one_print("max qbs") + ", " + one_print("unqomp"))
            if not is_snd:
                print("example, " + one_print_col_name("min_qbs") + ", " + one_print_col_name("075_qbs") + ", " + one_print_col_name("05_qbs"), file = f)# + ", " + one_print_col_name("025_qbs")+ ", " + one_print_col_name("max_qbs") + ", " + one_print_col_name("unqomp"), file = f)
            else:
                print("example, " + one_print_col_name("025_qbs")+ ", " + one_print_col_name("max_qbs") + ", " + one_print_col_name("unqomp"), file = f)
            gets_vals_one_example_app('adder', 100, 101, 1, identity, 0, False, 100, f, is_snd)
            gets_vals_one_example_app('dj', 100, 101, 1, identity, 0, False, 100, f, is_snd)
            gets_vals_one_example_app('grover', 10, 11, 2, identity, 0, False, 10, f, is_snd)
            gets_vals_one_example_app('intcomp', 100, 101, 1, identity, 1, False, 100, f, is_snd)
            gets_vals_one_example_app('mcry', 200, 201, 1, identity, 0, False, 200, f, is_snd)
            gets_vals_one_example_app('mcx', 200, 201, 1, identity, 0, False, 200, f, is_snd)
            gets_vals_one_example_app('mult', 16, 17, 1, identity, 0, True, 16, f, is_snd)
            gets_vals_one_example_app('plr', 40, 41, 3, identity, 4, False, 40, f, is_snd)
            gets_vals_one_example_app('polypr', 10, 11, 1, identity, 0, False, 10, f, is_snd)
            gets_vals_one_example_app('wa', 20, 21, 1, identity, 1, True, 20, f, is_snd)

def gets_timings_plots():
    makes_plot_timings_cleaner('adder', 100, 101, 1, identity, 100, True)
    makes_plot_timings_cleaner('grover', 10, 11, 2, identity, 10, True)
    makes_plot_timings_cleaner('dj', 100, 101, 1, identity, 100, True)
    makes_plot_timings_cleaner('mcx', 200, 201, 1, identity, 200, True)
    makes_plot_timings_cleaner('mcry', 200, 201, 1, identity, 200, True)
    makes_plot_timings_cleaner('polypr', 10, 11, 1, identity, 10, True)

    makes_plot_timings_cleaner('intcomp', 100, 101, 1, identity, 100, True)
    makes_plot_timings_cleaner('plr', 40, 41, 3, identity, 40, True)
    makes_plot_timings_cleaner('wa', 20, 21, 1, identity, 20, False)

    makes_plot_timings_cleaner('mult', 16, 17, 1, identity, 16, False)
    
gets_vals_all_examples()
gets_vals_all_examples_appendix()
makes_trade_offs_plots_one_file()
gets_timings_plots()