from reqomp.converter import ConverterCircuitGraph
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from reqomp.circuit_graph import CircuitGraph, Node
from qiskit import QuantumCircuit
from reqomp.graph_uncomputation import hierarchical_uncomputation


from qiskit.circuit import Qubit, QuantumRegister, QuantumCircuit, Gate, AncillaRegister

class AncillaGate:
    def __init__(self, gate, nb_ancillas = 0, extra_qfree = []):
        self._gate = gate
        self._nb_ancillas = nb_ancillas
        self._extra_qfree = extra_qfree

class AncillaCircuit(QuantumCircuit):
    # Mostly delegates to circuit, except for mcx, mcry where we use our custom implementation
    # plus allocates ancillas for gates
    def __init__(self, *regs, name = None):
        self._nb_ancillas = 0
        self._extra_qfree_gates = [] # records custom gates to consider qfree when uncomputing
        self._ancillas_list = []
        if isinstance(regs[0], int):
            QuantumCircuit.__init__(self, regs[0], name = name)
        elif isinstance(regs[0], AncillaRegister):
            QuantumCircuit.__init__(self, regs[0], name = name)
            self._nb_ancillas += len(regs[0][:])
            self._ancillas_list.append(regs[0])
        else:
            QuantumCircuit.__init__(self, regs[0], name = name)
        for reg in regs[1:]:
            self.add_register(reg)

    def append(self, instruction, qargs = None, cargs = None):
        if isinstance(instruction, AncillaGate) and instruction._nb_ancillas > 0:
            # adding ancillas as single registers would make linking more efficient, check if necessary?
            anc = AncillaRegister(instruction._nb_ancillas)
            self.add_register(anc) # updates nb ancillas
            assert cargs is None
            for qf in instruction._extra_qfree:
                if not qf in self._extra_qfree_gates:
                    self._extra_qfree_gates.append(qf)
            QuantumCircuit.append(self, instruction._gate, [*qargs, *anc[:]])
        elif isinstance(instruction, AncillaGate):
            for qf in instruction._extra_qfree:
                if not qf in self._extra_qfree_gates:
                    self._extra_qfree_gates.append(qf)
            QuantumCircuit.append(self, instruction._gate, qargs)
        else:
            QuantumCircuit.append(self, instruction, qargs, cargs)

    def add_register(self, reg):
        QuantumCircuit.add_register(self, reg)
        if isinstance(reg, AncillaRegister):
            self._ancillas_list.append(reg)
            self._nb_ancillas += len(reg[:])

    def new_ancilla_register(self, num_qubits, name = None):
        a = AncillaRegister(num_qubits, name)
        self.add_register(a)
        return a

    def to_ancilla_gate(self):
        # self should have registers in the following order: first ctrls, then target then ancillas
        gate = self.to_gate()
        extra_qfree_gates = self._extra_qfree_gates
        return AncillaGate(gate, self._nb_ancillas, extra_qfree_gates)

    def addQfreeGate(self, gate):
        self._extra_qfree_gates.append(gate)

    def mcx(self, ctrls, target):
        from reqomp.examples.mcx import makeMCX
        n = len(ctrls[:])
        mcx_gate = makeMCX(n).to_ancilla_gate()
        self.append(mcx_gate, [*ctrls[:], target])

    def mcry(self, rot_coeff, ctrls, target):
        from reqomp.examples.mcx import makeMCRY
        n = len(ctrls[:])
        mcry_gate = makeMCRY(rot_coeff, n).to_ancilla_gate()
        self.append(mcry_gate, [*ctrls[:], target])

    def uncompute(self, nbQbs: int, force_lazy = False):
        dag: DAGCircuit = circuit_to_dag(self)
        conv = ConverterCircuitGraph()
        graph: CircuitGraph = conv.dagToDepGraph(dag)
        graph_u: CircuitGraph = hierarchical_uncomputation(graph, nbQbs, force_lazy= force_lazy)
        dag_u: DAGCircuit = conv.depGraphToDag(graph_u, self.qubits)
        return dag_to_circuit(dag_u)