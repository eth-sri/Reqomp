import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import ZGate
from reqomp.ancilla_circuit import AncillaCircuit, AncillaGate

def makesOracle(i, n):
    # Creates the oracle finding exactly i on n qubits (+ 1 for target)(or its lowest n bits if i >= 2^n)
    # Could use some uncomputation...
    ctrls = QuantumRegister(n)
    target = QuantumRegister(1) 
    fcirc = AncillaCircuit(ctrls, target, name="oracle_" + str(i) + "_" + str(n))
    format_str = '{0:0' + str(n) + 'b}'
    binary_i = format_str.format(i)[::-1]
    for j in range(n):
        if binary_i[j] == '0':
            fcirc.x(ctrls[j])
    fcirc.mcx(ctrls[:], target[0])
    for j in range(n):
        if binary_i[j] == '0':
            fcirc.x(ctrls[j])
    return fcirc.to_ancilla_gate()
    
def makesGroverCircuit(n, oracle = None, nb_sols = 1):
    # grover circuit on n qubits, without measurements as uncomp cannot deal with that yet
    nbIter = int(np.floor(np.pi / 4.0 * np.sqrt(pow(2, n))))

    working_qubits = QuantumRegister(n, name = 'r')
    phase_qubit = QuantumRegister(1, name = 'p')
    circ = AncillaCircuit(working_qubits, phase_qubit)
    circ.x(phase_qubit[0])
    circ.h(phase_qubit[0])

    circ.h(working_qubits)
        
    for l in range(nbIter):
        if oracle is not None:
            circ.append(oracle, [*working_qubits[:], phase_qubit[0]])
        else:
            circ.mcx(working_qubits, phase_qubit)

        #Grover diffusion operator
        circ.h(working_qubits)
        circ.x(working_qubits)
        circ.h(working_qubits[-1])
        circ.mcx(working_qubits[:-1], working_qubits[-1])
        circ.h(working_qubits[-1])
        circ.x(working_qubits)
        
        circ.h(working_qubits)

    # bring the phase qubit back to 0, we can't uncompute it, as it went through cz, non qfree -> no need to uncomp it, Qiskit doesn't 
    #circ.h(phase_qubit[0])
    #circ.x(phase_qubit)
    return (circ, working_qubits)

def handbuiltQiskitGrover(i, n):
    # grover circuit on n qubits, without measurements as uncomp cannot deal with that yet
    nbIter = int(np.floor(np.pi / 4.0 * np.sqrt(pow(2, n))))

    working_qubits = QuantumRegister(n, name = 'r')
    phase_qubit = QuantumRegister(1, name = 'p')
    ancilla_qubits = QuantumRegister(n - 2, name = "ancilla")
    circ = QuantumCircuit(working_qubits, phase_qubit, ancilla_qubits)
    circ.x(phase_qubit[0])
    circ.h(phase_qubit[0])

    circ.h(working_qubits)
        
    for l in range(nbIter):
        format_str = '{0:0' + str(n) + 'b}'
        binary_i = format_str.format(i)[::-1]
        #print("binary " + str(binary_i))
        for j in range(n):
            if binary_i[j] == '0':
                circ.x(working_qubits[j])
        circ.mcx(working_qubits[:], phase_qubit[0], ancilla_qubits, mode = 'v-chain')
        for j in range(n):
            if binary_i[j] == '0':
                circ.x(working_qubits[j])

        #Grover diffusion operator
        circ.h(working_qubits)
        circ.x(working_qubits)
        circ.h(working_qubits[-1])
        circ.mcx(working_qubits[:-1], working_qubits[-1], ancilla_qubits, mode='v-chain')
        circ.h(working_qubits[-1])
        circ.x(working_qubits)
        
        circ.h(working_qubits)

    # bring the phase qubit back to 0, we can't uncompute it, as it went through cz, non qfree -> no need to uncomp it, Qiskit doesn't 
    #circ.h(phase_qubit[0])
    #circ.x(phase_qubit)
    return (circ, working_qubits)