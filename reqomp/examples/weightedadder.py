# Adapted from https://qiskit.org/documentation/_modules/qiskit/circuit/library/arithmetic/weighted_adder.html#WeightedAdder
from qiskit.circuit import QuantumRegister, QuantumCircuit, AncillaRegister
import numpy as np

from reqomp.ancilla_circuit import AncillaCircuit

def makeWeightedAdder(num_state_qubits, weights):
    # Straightforward implementation using Unqomp, uses less gates but more qubits
    num_sum_qubits = int(np.floor(np.log2(sum(weights))) + 1) if sum(weights) > 0 else 1 
    # The number of sum qubits in the circuit
    num_carry_qubits = num_sum_qubits -1
    # The number of carry qubits required to compute the sum.

    for i, weight in enumerate(weights):
        if not np.isclose(weight, np.round(weight)):
            raise ValueError('Non-integer weights are not supported!')
        weights[i] = np.round(weight)

    num_result_qubits = num_state_qubits + num_sum_qubits

    qr_state = QuantumRegister(num_state_qubits, name = 'state')
    qr_sum = QuantumRegister(num_sum_qubits, name = 'sum')
    circuit = AncillaCircuit(qr_state, qr_sum)
    #print(num_state_qubits)
    #print(num_sum_qubits)

    # loop over state qubits and corresponding weights
    for i, weight in enumerate(weights):
        # only act if non-trivial weight
        if np.isclose(weight, 0):
            continue

        # get state control qubit
        q_state = qr_state[i]

        # get bit representation of current weight
        weight_binary = '{0:b}'.format(int(weight)).rjust(num_sum_qubits, '0')[::-1]
        #print("carry qb" + str(num_carry_qubits))
        if num_carry_qubits > 0:
            qr_carry = circuit.new_ancilla_register(num_carry_qubits, name = "anccarryite" + str(i))

        # loop over bits of current weight and add them to sum and carry registers
        for j, bit in enumerate(weight_binary):
            if bit == '1':
                if num_sum_qubits == 1:
                    circuit.cx(q_state, qr_sum[j])
                elif j == 0:
                    # compute (q_sum[0] + 1) into (q_sum[0], q_carry[0])
                    # - controlled by q_state[i]
                    circuit.ccx(q_state, qr_sum[j], qr_carry[j])
                    circuit.cx(q_state, qr_sum[j])
                elif j == num_sum_qubits - 1:
                    # compute (q_sum[j] + q_carry[j-1] + 1) into (q_sum[j])
                    # - controlled by q_state[i] / last qubit,
                    # no carry needed by construction
                    circuit.cx(q_state, qr_sum[j])
                    circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
                else:
                    # compute (q_sum[j] + q_carry[j-1] + 1) into (q_sum[j], q_carry[j])
                    # - controlled by q_state[i]
                    circuit.x(qr_sum[j])
                    circuit.x(qr_carry[j - 1])
                    circuit.mcx([q_state, qr_sum[j], qr_carry[j - 1]], qr_carry[j])
                    circuit.x(qr_sum[j])
                    circuit.x(qr_carry[j - 1])
                    circuit.cx(q_state, qr_carry[j])
                    circuit.cx(q_state, qr_sum[j])
                    circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
            else:
                if num_sum_qubits == 1:
                    pass  # nothing to do, since nothing to add
                elif j == 0:
                    pass  # nothing to do, since nothing to add
                elif j == num_sum_qubits-1:
                    # compute (q_sum[j] + q_carry[j-1]) into (q_sum[j])
                    # - controlled by q_state[i] / last qubit,
                    # no carry needed by construction
                    circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
                else:
                    # compute (q_sum[j] + q_carry[j-1]) into (q_sum[j], q_carry[j])
                    # - controlled by q_state[i]
                    circuit.mcx([q_state, qr_sum[j], qr_carry[j - 1]], qr_carry[j])
                    circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
    return circuit

