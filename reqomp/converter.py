from qiskit.circuit import Instruction, Qubit, QuantumRegister, AncillaQubit # type: ignore
from qiskit.circuit.library import RCCXGate, IGate
from qiskit.converters import circuit_to_dag, dag_to_circuit # type: ignore
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGInNode, DAGOpNode, DAGOutNode # type: ignore
from reqomp.circuit_graph import CircuitGraph, Edge, Node, InitGate
from typing import List, Dict

class ConverterCircuitGraph:
    known_qfree_gates = ['ccx', 'cnot', 'cx', 'i', 'id', 'iden', 'mct', 'mcx', 'mcx_gray', 'toffoli', 'x']
    known_non_qfree_gates = ['ch', 'crx', 'cry', 'crz', 'cu1', 'cu2', 'cu3', 'cy', 'cz', 'h', 'mcrx', 'mcry', 'mcrz', 'mcu1', 'r', 'rcccx', 'rccx', 'rx', 'ry', 'rz', 's', 'sdg', 't', 'tdg', 'u', 'u1', 'u2', 'u3', 'ucrx', 'ucry', 'ucrz', 'ucx', 'ucy', 'ucz', 'y', 'z']
    # left out on purpose: cswap, dcx, fredkin, iswap, mcmt, ms, rxx, ryy, rzx, rzz, swap, as those have two targets

    def __init__(self, extra_gates: List[Instruction] = []):
        # list of gates names to add to known_gates + bool if qfree (arguments have to be of the form (ctrl*, target))
        self.extra_qfree_gates: List[str] = []
        self.extra_non_qfree_gates: List[str] = []
        for (gate, is_qfree) in extra_gates:
            if is_qfree:
                self.extra_qfree_gates.append(gate.name)
            else:
                self.extra_non_qfree_gates.append(gate.name)
        pass
       

    def _isKnownInstruction(self, instruction: Instruction) -> bool:
        if instruction.name in self.extra_qfree_gates or instruction.name in self.extra_non_qfree_gates:
            return True
        if instruction.name in ConverterCircuitGraph.known_qfree_gates or instruction.name in ConverterCircuitGraph.known_non_qfree_gates:
            return True
        return False

    def _decomposeDAGToKnownGates(self, dag: DAGCircuit) -> DAGCircuit:
        unseen_gates = True
        while(unseen_gates):
            unseen_gates = False
            old_dag_nodes = dag.op_nodes()
            for node in old_dag_nodes:
                if not self._isKnownInstruction(node.op):
                    unseen_gates = True
                    assert self._decomposeNodeOnDAG(node, dag), "Could not decompose DAG to known gates" + str(node.op)
        return dag

    def _simplUGates(self, dag: DAGCircuit) -> DAGCircuit:
        dag_nodes = dag.op_nodes()
        for node in dag_nodes:
            if node.op.name == 'u1' and node.op.params[0] == 0:
                node.op = IGate()
        for node in dag_nodes:
            if node.op.name == 'u' and node.op.params[0] == 0 and node.op.params[1] == 0 and node.op.params[2] == 0.0: #detects Id gates that have been implemented with u
                node.op = IGate()
        return dag

    def _decomposeNodeOnDAG(self, node: DAGNode, dag: DAGCircuit) -> bool:
        # we cannot use qiskit.transpiler.passes.Decompose, since any custom gate is of type Instruction, and 
        # Decompose checks the type, so no way to decompose only those custom gates
        if not node.op.definition:
            print("Could not decompose node " + str(node)+ ", of operation " + str(node.op.name))
            return False
        node_decomposition = DAGCircuit()
        qreg = set()
        creg = set()
        for instruction in node.op.definition:
            for qubit in instruction[1]:
                if not qubit.register in qreg:
                    node_decomposition.add_qreg(qubit.register)
                    qreg.add(qubit.register)
            for cbit in instruction[2]:
                 if not cbit.register in creg:
                    node_decomposition.add_creg(cbit.register)
                    creg.add(cbit.register)
        
        for instruction in node.op.definition:
            node_decomposition.apply_operation_back(*instruction)
        
        dag.substitute_node_with_dag(node, node_decomposition)
        return True

    def isQfree(operation: Instruction, circ_graph: CircuitGraph) -> bool:
        if operation.name in ConverterCircuitGraph.known_qfree_gates or operation in circ_graph.extra_qfree_gates:
            return True
        return False

    def dagToDepGraph(self, dag: DAGCircuit) -> CircuitGraph:
        # we first remove "empty" gates (can occur if we composed with an empty circuit, and they cause problems later)
        to_remove = []
        for n in dag.op_nodes():
            if n.op.definition is None and len(n.op.decompositions) == 0: # this is an empty gate
                to_remove.append(n)
        for n in to_remove:
            dag.remove_op_node(n)
        #We first decompose all the gates into gates we know how to deal with
        dag = self._decomposeDAGToKnownGates(dag)
        dag = self._simplUGates(dag)

        circ_g = CircuitGraph(self.extra_qfree_gates, self.extra_non_qfree_gates)
        latest_node_on_wire = {}

        for dag_node in dag.topological_nodes():
            if isinstance(dag_node, DAGInNode):
                # this is the beginning of a wire, green node in dag drawings
                is_anc = isinstance(dag_node.wire.register[dag_node.wire.index], AncillaQubit)
                qubit = dag_node.wire.register[dag_node.wire.index]
                node = Node(dag_node.wire.register[dag_node.wire.index], is_anc, True, InitGate(), qubit.register.name + "_" + str(qubit.index), 0, 0)
                latest_node_on_wire[node.variable_name] = node
                circ_g.addNode(node)
            elif isinstance(dag_node, DAGOpNode):
                modified_qubit = dag_node.qargs[-1]
                wire_name = Node.qubitName(modified_qubit)
                assert self._isKnownInstruction(dag_node.op)
                assert latest_node_on_wire.get(wire_name) is not None
                previous_node_on_wire = latest_node_on_wire[wire_name]
                ctrlNodes = [latest_node_on_wire[Node.qubitName(qarg)] for qarg in dag_node.qargs[:-1]] #assumes last element is always the target qubit
                value_id = 0
                is_qfree = False
                if ConverterCircuitGraph.isQfree(dag_node.op, circ_g):
                    value_id = circ_g.getValueId(previous_node_on_wire, ctrlNodes, dag_node.op, True)
                    is_qfree = True
                else:
                    value_id = circ_g.getValueId(previous_node_on_wire, ctrlNodes, dag_node.op, False)
                copy_id = circ_g.newCopyId(wire_name, value_id)
                node = Node(modified_qubit, previous_node_on_wire.is_ancilla, is_qfree, dag_node.op, modified_qubit.register.name + "_" + str(modified_qubit.index), value_id, copy_id)
                circ_g.addNode(node)
                circ_g.connectTargetNodes(previous_node_on_wire, node)
                latest_node_on_wire[node.variable_name] = node
                id_ctrl = 0
                for ctrl_node in ctrlNodes:
                    circ_g.connectControlNodes(ctrl_node, node, id_ctrl)
                    id_ctrl += 1
            elif not isinstance(dag_node, DAGOutNode):
                print("Neither init nor op node nor out node" + str(dag_node) + " " + str(dag_node.type))  
        return circ_g

    def depGraphToDag(self, circ_graph: CircuitGraph, qubits_in_order: List[Qubit]) -> DAGCircuit:
        circ_graph.graph_checks()
        # first add the registers in the proper order
        dag = DAGCircuit()
        already_added: Dict[QuantumRegister, bool] = {} # registers may yield mutliple wires, we don't want to add them multiple times
        for qubit in qubits_in_order:
            if already_added.get(qubit.register) is None and not isinstance(qubit, AncillaQubit):
                    already_added[qubit.register] = True
                    dag.add_qreg(qubit.register)

        lowestSuccessorValueId = self._getLowestSuccessorValueId(circ_graph) # for each node, the lowest value id on its target chain after itself

        for dep_node in circ_graph.nodesInTopologicalOrder():
            if dep_node.gate.name == 'init':
                if already_added.get(dep_node.qubit.register) is None:
                    already_added[dep_node.qubit.register] = True
                    dag.add_qreg(dep_node.qubit.register)
            elif dep_node.gate.name != 'deallocate': #nothing to do for deallocate nodes
                #get the arguments of the gate: again assuming ctrls first, target last
                ctrl_edges = [c for c  in dep_node.edges_in if c.type == 'c']
                def get_order(edge: Edge):
                    return edge.id_ctrl
                ctrl_reg = [ctrl_edge.node_from.qubit for ctrl_edge in sorted(ctrl_edges, key=get_order)]
                assert dep_node.target_edge_in.node_from.qubit == dep_node.qubit # type: ignore
                target_reg = dep_node.qubit
                ctrl_reg.append(target_reg)
                # can we replace CCX with RCCX? yes if its an uncomputation node (ie with an odd copy id) or if it's uncomputed later (ie has successors with lower value_id)
                if dep_node.copy_id % 2 == 1 or lowestSuccessorValueId[dep_node] < dep_node.value_id:
                    dep_node.gate = _replaceCCXs(dep_node.gate)
                dag.apply_operation_back(dep_node.gate, ctrl_reg)
        return dag

    def _getLowestSuccessorValueId(self, circ_g: CircuitGraph) -> Dict[Node, int]:
        lowestSuccessorValueId: Dict[Node, int] = {}
        def rec_find_id(node):
            if lowestSuccessorValueId.get(node) is not None:
                return lowestSuccessorValueId.get(node)
            if node.target_edge_out is None:
                lowestSuccessorValueId[node] = node.value_id
                return node.value_id
            res = min(rec_find_id(node.target_edge_out.node_to), node.value_id)
            lowestSuccessorValueId[node] = res
            return res
        for var_name in circ_g.nodes:
            for cid in range(len(circ_g.nodes[var_name][0])):
                rec_find_id(circ_g.nodes[var_name][0][cid])
        return lowestSuccessorValueId

def _replaceCCXs(gate: Instruction) -> Instruction:
        if gate.name == 'ccx':
            return RCCXGate()
        if gate.definition is None:
            return gate

        gate.definition._data = [(_replaceCCXs(inst), qargs, cargs)
                                         for inst, qargs, cargs in gate.definition]
        return gate

        
