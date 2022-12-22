from qiskit.circuit import QuantumRegister, QuantumCircuit, AncillaRegister
from qiskit.circuit.library.standard_gates import HGate
from reqomp.ancilla_circuit import AncillaCircuit

x = QuantumRegister(1, name = 'x')
y = QuantumRegister(1, name = 'y')
b = QuantumRegister(1, name = 'b')
c = AncillaRegister(1, name = 'c')

o = QuantumRegister(1, 'o')
p = QuantumRegister(1, 'p')
q = QuantumRegister(1, 'q')
r = QuantumRegister(1, 'r')
t = QuantumRegister(1, 't')
a = AncillaRegister(1, 'a')
b = AncillaRegister(1, 'b')
c = AncillaRegister(1, 'c')

circ = AncillaCircuit(o, p, q, r, a, b, c, t)

circ.ccx(o, p, a) # all gates can be used on an AncillaCircuit directly
circ.ccx(q, a, b)
circ.ccx(r, b, c)
circ.append(HGate().control(1), [c, t]) 

print(circ) # AncillaCircuit contains no uncomputation

# Uncomputation is added using the provided number of ancilla qubits, here 3. Uncomputed CCX gates are further replaced by the Margolus RCCX gate. The AncillaCircuit is converted to a qiskit QuantumCircuit, which does not distinguish ancillas from other qubits.
# Here 3 ancilla qubits are used, so all ancilla variables can be computed and uncomputed exactly once.
circ3 = circ.uncompute(3)

print(circ3)

# Now only 2 ancilla qubits are used, so ancilla variable a is uncomputed early and recomputed, to allow ancilla variable c to use the same ancilla qubit a4.
circ2 = circ.uncompute(2)
print(circ2)


from qiskit import QuantumRegister, AncillaRegister
from reqomp.ancilla_circuit import AncillaCircuit

x = QuantumRegister(1, name = 'x')
y = QuantumRegister(1, name = 'y')
b = QuantumRegister(1, name = 'b')
c = AncillaRegister(1, name = 'c')

circ = AncillaCircuit(x, y, b, c)
circ.ccx(b, x, c)
circ.cx(b, x)
circ.cx(c, y)

gate = circ.to_ancilla_gate() # control qubits should come first, then target qubit then ancillae.

z = QuantumRegister(1, name = 'z')
t = QuantumRegister(1, name = 't')
d = QuantumRegister(1, name = 'd')
circ2 = AncillaCircuit(z, t, d)

# append gate to circ2, adding d to (z, t), the ancilla c is appended automatically
circ2.append(gate, [z, t, d])
# appending it again, a new ancilla is added again
circ2.append(gate, [z, t, d])

print(circ2)

circ2 = circ2.uncompute(1) # the two ancillas are now allocated on the same qubit

print(circ2)
