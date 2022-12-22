import sys
import signal
from random import randint, seed, sample


sys.setrecursionlimit(20000)

# Files format:
# Recomp files:
    # time construction circuit
    # time conversion to dag
    # time conversion to dep graph
    # time uncomputation
    # time conversion to dag
    # time conversion to circuit
    # time decomposition
    # nb qbs; nb gates tot; nb cx for reqomp; circuit depth
# Qiskit files:
    #time construction with unqomp
    # time decomposition
    # nb qbs; nb gates tot; nb cx for qiskit; circuit depth
# Unqomp files:
    # time construction circuit
    # time unqomp all included
    # time decomposition
    # nb qbs; nb gates tot; nb cx for reqomp; circuit depth

#from https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
from functools import wraps
from time import time
from typing import Tuple, Union
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.compiler import transpile

time_out_cst = 3000000

def timing(f, file, timeout_s):
    @wraps(f)
    def wrap(*args, **kw):
        def handler(signum, frame):
            raise TimeoutError()
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_s)
        try:
            ts = time()
            result = f(*args, **kw)
            te = time()
            print(te-ts, file = file)
            return (True, result)
        except TimeoutError:
            print("timed out in ", f.__name__)
            print("TO", file = file)
            return (False, {})
        finally:
            signal.alarm(0)
    return wrap

def count_ops_c(circuit: QuantumCircuit):
    basis_gates = ['id', 'u', 'cx']
    qc_decomp = transpile(circuit, basis_gates=basis_gates, optimization_level= 0)
    qc_trans = qc_decomp.count_ops()
    print(qc_trans)
    final_count = {'cx': 0, 'u':0}
    if qc_trans.get('cx') is not None:
        final_count['cx'] = qc_trans['cx']
    if qc_trans.get('u') is not None:
        final_count['u'] = qc_trans['u']
    if qc_trans.get('id') is not None:
        final_count['u'] += qc_trans['id']
    final_count['d'] = qc_decomp.depth()
    return final_count

def timed_uncomputation(circuit, nb_qubits: int, file):
    try:
        circuit_to_dag_t = timing(circuit_to_dag, file, time_out_cst)
        (finished, dag) = circuit_to_dag_t(circuit)
        if not finished:
            return False
        conv = ConverterCircuitGraph()
        dag_to_dep_graph_t = timing(conv.dagToDepGraph, file, time_out_cst)
        (finished, graph) = dag_to_dep_graph_t(dag)
        if not finished:
            return False
        uncompute_t = timing(hierarchical_uncomputation, file, time_out_cst)
        (finished,graph_u) = uncompute_t(graph, nb_qubits)
        if not finished:
            return False
        dep_graph_to_dag_t = timing(conv.depGraphToDag, file, time_out_cst)
        (finished, dag_u) = dep_graph_to_dag_t(graph_u, circuit.qubits)
        if not finished:
            return False
        dag_to_circuit_t = timing(dag_to_circuit, file, time_out_cst)
        (finished,res) = dag_to_circuit_t(dag_u)
        if not finished:
            return False
        nb_qb = res.num_qubits
        decompo_t = timing(count_ops_c, file, time_out_cst)
        (finished, nb_gates) = decompo_t(res)
        if not finished:
            return False # print nothing, which will be detected by plotting
        if nb_gates.get('cx') is None:
            nb_gates['cx'] = 0
        if nb_gates.get('u') is None:
            nb_gates['u'] = 0
        if nb_gates['d'] is None:
            nb_gates['d'] = 0
        print(str(nb_qubits) + ';' + str(nb_qb) + ' ; ' + str(nb_gates['cx'] + nb_gates['u']) + ' ; ' + str(nb_gates['cx']) + ' ; ' + str(nb_gates['d']), file=file)

        return False
    except NotEnoughAncillas:
        file.seek(0)
        file.truncate(0)
        print("N", file = file) #not enough ancillas
        return True

def timed_uncomputation_unqomp(circuit, file):
    try:
        uncomp_t = timing(circuit.circuitWithUncomputation, file, time_out_cst)
        (finished, res) = uncomp_t()
        if not finished:
            return
        nb_qb = res.num_qubits
        decompo_t = timing(count_ops_c, file, time_out_cst)
        (finished, nb_gates) = decompo_t(res)
        if not finished:
            return # print nothing, which will be detected by plotting
        if nb_gates.get('cx') is None:
            nb_gates['cx'] = 0
        if nb_gates.get('u') is None:
            nb_gates['u'] = 0
        if nb_gates['d'] is None:
            nb_gates['d'] = 0
        print(str(nb_qb) + ' ; ' + str(nb_gates['cx'] + nb_gates['u']) + ' ; ' + str(nb_gates['cx']) + ' ; ' + str(nb_gates['d']), file=file)
    except Exception:
        file.seek(0)
        file.truncate(0)
        print("X", file = file) #error to investigate :(

def grover(n_min, n_max, n_step, unqomp_runs):
    if not unqomp_runs:
        from reqomp.examples.grover import makesGroverCircuit, handbuiltQiskitGrover
    else:
        import unqomp.examples.grover as Ugrover

    for n in range(n_min, n_max, n_step):
        if not unqomp_runs:
            for nb_anc in range(n, 0, -1):
                print("grover", n)
                filename = "evaluation_results/grover_" + str(n) + "_" + str(nb_anc)
                file = open(filename, "w")
                timed_dj = timing(makesGroverCircuit, file, time_out_cst)
                (finished, (circuit, reg)) = timed_dj(n)
                notenoughanc = timed_uncomputation(circuit, nb_anc, file)
                file.close()
                if notenoughanc:
                    break
        else:
            filename = "evaluation_results/grover_" + str(n) + "_u"
            file = open(filename, "w")
            timed_constr = timing(Ugrover.makesGroverCircuit, file, time_out_cst)
            (finished, (circuit_u, reg)) = timed_constr(n)
            timed_uncomputation_unqomp(circuit_u, file)
            file.close()
            

def adder(n_min, n_max, n_step, unqomp_runs):
    if not unqomp_runs:
        from reqomp.examples.adder import makesAdder, makesCirqAdder
    else:
        import unqomp.examples.adder as Uadder

    for n in range(n_min, n_max, n_step):
        if not unqomp_runs:
            for nb_anc in range(n, 0, -1):
                print("adder", n)
                filename = "evaluation_results/adder_" + str(n) + "_" + str(nb_anc)
                file = open(filename, "w")
                timed_dj = timing(makesAdder, file, time_out_cst)
                (finished, circuit) = timed_dj(n)
                notenoughanc = timed_uncomputation(circuit, nb_anc, file)
                file.close()
                if notenoughanc:
                    break
        else:
            filename = "evaluation_results/adder_" + str(n) + "_u"
            file = open(filename, "w")
            timed_constr = timing(Uadder.makesAdder, file, time_out_cst)
            (finished, circuit_u) = timed_constr(n)
            timed_uncomputation_unqomp(circuit_u, file)
            file.close()

def mult(n_min, n_max, n_step, unqomp_runs):
    if not unqomp_runs:
        from reqomp.examples.adder import makesMult, makesCirqMult
    else:
        import unqomp.examples.adder as Uadder

    for n in range(n_min, n_max, n_step):
        if not unqomp_runs:
            print("mult", n)
            filename = "evaluation_results/mult_" + str(n)
            file = open(filename, "w")
            timed_dj = timing(makesMult, file, time_out_cst)
            (finished, circuit) = timed_dj(n)
            notenoughanc = timed_uncomputation(circuit, n * 4, file)
            file.close()
        else:
            filename = "evaluation_results/mult_" + str(n) + "_u"
            file = open(filename, "w")
            timed_constr = timing(Uadder.makesMult, file, time_out_cst)
            (finished, circuit_u) = timed_constr(n)
            timed_uncomputation_unqomp(circuit_u, file)
            file.close()

def dj(n_min, n_max, n_step, unqomp_runs):
    if not unqomp_runs:
        from reqomp.examples.deutschjozsa import makesDJ, QiskitDJ
    else:
        from unqomp.examples.deutschjozsa import makesDJ as Udj

    for n in range(n_min, n_max, n_step):
        if not unqomp_runs:
            for nb_anc in range(n, 0, -1):
                print("dj", n)
                filename = "evaluation_results/dj_" + str(n) + "_" + str(nb_anc)
                file = open(filename, "w")
                timed_dj = timing(makesDJ, file, time_out_cst)
                (finished, (circuit, reg)) = timed_dj(n)
                notenoughanc = timed_uncomputation(circuit, nb_anc, file)
                file.close()
                if notenoughanc:
                    break
        else:
            filename = "evaluation_results/dj_" + str(n) + "_u"
            file = open(filename, "w")
            timed_constr = timing(Udj, file, time_out_cst)
            (finished, (circuit_u, reg)) = timed_constr(n)
            timed_uncomputation_unqomp(circuit_u, file)
            file.close()

def integercomparator(n_min, n_max, n_step, unqomp_runs):
    if not unqomp_runs:
        from reqomp.examples.intergercomparator import makeIntegerComparator
    else:
        import unqomp.examples.intergercomparator as UIc

    # 10 randint, within [0, 1 << n] both included, in file name

    for n in range(n_min, n_max, n_step):
        v = randint(0, 1 << n)
        if not unqomp_runs:
            for nb_anc in range(n, 0, -1):
                print("intcomp", n)
                filename = "evaluation_results/intcomp_" + str(n) + "_" + str(nb_anc)
                file = open(filename, "w")
                timed_dj = timing(makeIntegerComparator, file, time_out_cst)
                (finished, circuit) = timed_dj(n, v)
                notenoughanc = timed_uncomputation(circuit, nb_anc, file)
                print(v, file = file)
                file.close()
                if notenoughanc:
                    break
        else:
            filename = "evaluation_results/intcomp_" + str(n) + "_u"
            file = open(filename, "w")
            timed_constr = timing(UIc.makeIntegerComparator, file, time_out_cst)
            (finished, circuit_u) = timed_constr(n, v)
            timed_uncomputation_unqomp(circuit_u, file)
            print(v, file = file)
            file.close()

def mcx(n_min, n_max, n_step, unqomp_runs):
    if not unqomp_runs:
        from reqomp.ancilla_circuit import AncillaCircuit
    
    from qiskit import QuantumRegister, QuantumCircuit

    def build_mcx_circ(n_q):
        c = QuantumRegister(n_q)
        t = QuantumRegister(1)
        circuit = AncillaCircuit(c, t)
        circuit.mcx(c, t)
        return circuit

    def build_mcx_u(n_q):
        c = QuantumRegister(n_q)
        t = QuantumRegister(1)
        circuit = unqompAlloc.AncillaCircuit(c, t)
        circuit.mcx(c, t)
        return circuit

    for n in range(n_min, n_max, n_step):
        if not unqomp_runs:
            for nb_anc in range(n, 0, -1):
                print("mcx", n)
                filename = "evaluation_results/mcx_" + str(n) + "_" + str(nb_anc)
                file = open(filename, "w")
                timed_dj = timing(build_mcx_circ, file, time_out_cst)
                (finished, circuit) = timed_dj(n)
                notenoughanc = timed_uncomputation(circuit, nb_anc, file)
                file.close()
                if notenoughanc:
                    break
        else:
            filename = "evaluation_results/mcx_" + str(n) + "_u"
            file = open(filename, "w")
            timed_constr = timing(build_mcx_u, file, time_out_cst)
            (finished, circuit_u) = timed_constr(n)
            timed_uncomputation_unqomp(circuit_u, file)
            file.close()

def mcry(n_min, n_max, n_step, unqomp_runs):
    if not unqomp_runs:
        from reqomp.ancilla_circuit import AncillaCircuit
    from qiskit import QuantumRegister, QuantumCircuit

    def build_mcry_circ(n_q):
        c = QuantumRegister(n_q)
        t = QuantumRegister(1)
        circuit = AncillaCircuit(c, t)
        circuit.mcry(4, c, t)
        return circuit
    
    def build_mcry_u(n_q):
        c = QuantumRegister(n_q)
        t = QuantumRegister(1)
        circuit = unqompAlloc.AncillaCircuit(c, t)
        circuit.mcry(4, c, t)
        return circuit

    for n in range(n_min, n_max, n_step):
        if not unqomp_runs:
            for nb_anc in range(n, 0, -1):
                print("mcry", n)
                filename = "evaluation_results/mcry_" + str(n) + "_" + str(nb_anc)
                file = open(filename, "w")
                timed_dj = timing(build_mcry_circ, file, time_out_cst)
                (finished, circuit) = timed_dj(n)
                notenoughanc = timed_uncomputation(circuit, nb_anc, file)
                file.close()
                if notenoughanc:
                    break
        else:
            filename = "evaluation_results/mcry_" + str(n) + "_u"
            file = open(filename, "w")
            timed_constr = timing(build_mcry_u, file, time_out_cst)
            (finished, circuit_u) = timed_constr(n)
            timed_uncomputation_unqomp(circuit_u, file)
            file.close()

def piecewiselinearrot(n_min, n_max, n_step, unqomp_runs):
    if not unqomp_runs:
        from reqomp.examples.piecewiselinrot import makesPLR
    else:
        import unqomp.examples.piecewiselinrot as Uplr

    # random nb of breakpoints in [0, 100], then random breakpoints, then random slopes then random offsets

    for n in range(n_min, n_max, n_step):
            nb_breaks = randint(2, min(1 << n, 100)) # number of breakpoints, >= 2 bc Qiskit dies for 1
            breakpoints = sample(range(0, 1 << n), nb_breaks)
            breakpoints.sort()
            slopes = [randint(0, 100) for i in range(nb_breaks)]
            offsets = [randint(0, 200) for i in range(nb_breaks)]
            print(nb_breaks)
            print(breakpoints)
            print(slopes)
            print(offsets)
            #print them out to file
            if not unqomp_runs:
                for nb_anc in range(n, 0, -1):
                    print("plr", n)
                    filename = "evaluation_results/plr_" + str(n) + "_" + str(nb_anc)
                    file = open(filename, "w")
                    timed_dj = timing(makesPLR, file, time_out_cst)
                    (finished, circuit) = timed_dj(n, breakpoints, slopes, offsets)
                    notenoughanc = timed_uncomputation(circuit, nb_anc, file)
                    print(nb_breaks, file = file)
                    print(breakpoints, file = file)
                    print(slopes, file = file)
                    print(offsets, file = file)
                    file.close()
                    if notenoughanc:
                        break
            else:
                filename = "evaluation_results/plr_" + str(n) + "_u"
                file = open(filename, "w")
                timed_constr = timing(Uplr.makesPLR, file, time_out_cst)
                (finished, circuit_u) = timed_constr(n, breakpoints, slopes, offsets)
                timed_uncomputation_unqomp(circuit_u, file)
                print(nb_breaks, file = file)
                print(breakpoints, file = file)
                print(slopes, file = file)
                print(offsets, file = file)
                file.close()

def polypaulirot(n_min, n_max, n_step, unqomp_runs):
    if not unqomp_runs:
        from reqomp.examples.polynomialpaulirot import makesPolyPauliRot
    else:
        import unqomp.examples.polynomialpaulirot as Uppr

    # coefficients don't matter
    coeffs = [2 for i in range(n_max)]
    check_semantics = True

    for n in range(n_min, n_max, n_step):
        if not unqomp_runs:
            for nb_anc in range(n, 0, -1):
                print("polypr", n)
                filename = "evaluation_results/polypr_" + str(n) + "_" + str(nb_anc)
                file = open(filename, "w")
                timed_dj = timing(makesPolyPauliRot, file, time_out_cst)
                (finished, circuit) = timed_dj(n, coeffs[:n])
                notenoughanc = timed_uncomputation(circuit, nb_anc, file)
                file.close()
        else:
            filename = "evaluation_results/polypr_" + str(n) + "_u"
            file = open(filename, "w")
            timed_constr = timing(Uppr.makesPolyPauliRot, file, time_out_cst)
            (finished, circuit_u) = timed_constr(n, coeffs[:n])
            timed_uncomputation_unqomp(circuit_u, file)
            file.close()

def weightedadder(n_min, n_max, n_step, unqomp_runs):
    if not unqomp_runs:
        from reqomp.examples.weightedadder import makeWeightedAdder
    else:   
        import unqomp.examples.weightedadder as Uwa

    for n in range(n_min, n_max, n_step):
            vals = [randint(0, 10) for j in range(n)]
            print("wa", n)
            if not unqomp_runs:
                filename = "evaluation_results/wa_" + str(n)
                file = open(filename, "w")
                timed_dj = timing(makeWeightedAdder, file, time_out_cst)
                (finished, circuit)  = timed_dj(n, vals)
                notenoughanc = timed_uncomputation(circuit, n * 4, file) # nb anc does not matter, it goes to lazy
                print(vals, file = file)
                file.close()
            else:
                filename = "evaluation_results/wa_" + str(n) + "_u"
                file = open(filename, "w")
                timed_constr = timing(Uwa.makeWeightedAdder, file, time_out_cst)
                (finished, circuit_u) = timed_constr(n, vals)
                timed_uncomputation_unqomp(circuit_u, file)
                print(vals, file = file)
                file.close()
        
def runall():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-u", "--unqomp",
                    action="store_true", dest="unqomp_runs", default=False,
                    help="run unqomp on benchmarks for comparison (and not reqomp)")
    parser.add_argument("-a", "--all",
                    action="store_true", dest="all_runs", default=False,
                    help="run all examples (not only the small ones)")

    args = parser.parse_args()
    unqomp_runs = args.unqomp_runs
    all_runs = args.all_runs

    if(unqomp_runs):
        global unqompAlloc
        import unqomp.ancillaallocation as unqompAlloc
    else:
        global AncillaCircuit, CircuitGraph, ConverterCircuitGraph, NotEnoughAncillas, hierarchical_uncomputation
        from reqomp.ancilla_circuit import AncillaCircuit
        from reqomp.circuit_graph import CircuitGraph
        from reqomp.converter import ConverterCircuitGraph
        from reqomp.graph_uncomputation import NotEnoughAncillas, hierarchical_uncomputation

    seed(2)

    adder(12, 13, 1, unqomp_runs)
    dj(10, 11, 1, unqomp_runs)
    grover(5, 6, 2, unqomp_runs)
    integercomparator(12, 13, 1, unqomp_runs)
    mcx(12, 13, 1, unqomp_runs)
    mcry(12, 13, 1, unqomp_runs)
    mult(5, 6, 1, unqomp_runs)
    piecewiselinearrot(6, 7, 3, unqomp_runs)
    polypaulirot(5, 6, 1, unqomp_runs)
    weightedadder(10, 11, 1, unqomp_runs)
    print("done small")

    if(all_runs):
        adder(100, 101, 1, unqomp_runs)
        dj(100, 101, 1, unqomp_runs)
        grover(10, 11, 2, unqomp_runs)
        integercomparator(100, 101, 1, unqomp_runs)
        mcx(200, 201, 1, unqomp_runs)
        mcry(200, 201, 1, unqomp_runs)
        mult(16, 17, 1, unqomp_runs)
        piecewiselinearrot(40, 41, 3, unqomp_runs)
        polypaulirot(10, 11, 1, unqomp_runs)
        weightedadder(20, 21, 1, unqomp_runs)
        print("done big")

runall()
