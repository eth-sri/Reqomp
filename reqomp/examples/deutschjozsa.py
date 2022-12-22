import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from reqomp.ancilla_circuit import AncillaCircuit, AncillaGate

def makesDJ(num_qubits, oracle_gate = None):
    #Builds the Deutsch Jozsa circuit for n + 1 qubits, finding the value 111...111
    var_reg = QuantumRegister(num_qubits, name = 'vals')
    out_reg = QuantumRegister(1, name = 'out')

    circ = AncillaCircuit(var_reg, out_reg)

    circ.h(var_reg)
    circ.x(out_reg)
    circ.h(out_reg)

    if oracle_gate:
        circ.append(oracle_gate, [*var_reg, out_reg[0]])
    else:
        circ.mcx(var_reg, out_reg)

    circ.h(var_reg)

    return (circ, var_reg)

def QiskitDJ(n):
    #Builds the Deutsch Jozsa circuit for n + 1 qubits, finding the value 111...111
    var_reg = QuantumRegister(n, name = 'vals')
    anc_reg = QuantumRegister(n - 2)
    out_reg = QuantumRegister(1, name = 'out')

    circ = QuantumCircuit(var_reg, out_reg, anc_reg)

    circ.h(var_reg)
    circ.x(out_reg)
    circ.h(out_reg)

    circ.mcx(var_reg, out_reg, anc_reg, mode='v-chain')

    circ.h(var_reg)

    return circ