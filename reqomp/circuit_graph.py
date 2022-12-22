from __future__ import annotations
from copy import deepcopy
from qiskit.circuit import Qubit, Instruction, AncillaQubit # type: ignore
from typing import Deque, List, Dict, Optional, Tuple
from collections import deque


class InitGate:
    def __init__(self):
        self.name = 'init'

class DeallocateGate:
    def __init__(self):
        self.name = 'deallocate'

class Edge:
    def __init__(self, node_from: Node, node_to: Node, edge_type: str, id_ctrl:int = 0):
        self.node_from = node_from
        self.node_to = node_to
        self.type = edge_type #'t' for target, 'c' for control, 'a' for availability = write after read
        self.id_ctrl = id_ctrl #if it's a ctrl edge, which one it is 0th, 1st, 2nd...)

    def __repr__(self):
        return "Edge: from({}), to({}), type({}), ctrl_id({})".format(self.node_from, self.node_to, self.type, self.id_ctrl)

class Node:
    def __init__(self, qubit: Qubit, is_anc: bool, is_qfr: bool, gate: Instruction, var_name: str, value_id: int = 0, copy_id: int = 0):
        self.qubit = qubit
        self.is_ancilla = is_anc
        self.is_qfree = is_qfr
        self.variable_name = var_name # if not None else qubit.register.name + "_" + str(qubit.index)
        self.gate = gate
        self.value_id = value_id
        self.copy_id = copy_id
        self.edges_in: List[Edge] = [] #no target edges in there
        self.edges_out: List[Edge] = [] #no target edges in there
        self.target_edge_in: Optional[Edge] = None
        self.target_edge_out: Optional[Edge] = None

    def __repr__(self) -> str:
        return "Node: var({}), vId({}), cId({})".format(self.variable_name, self.value_id, self.copy_id)

    def __gt__(self, node2: Node) -> bool: #order does not matter so much, we just want to break ties when comparing ancillaRegisters
        if self.qubit.register.name == node2.qubit.register.name:
            if self.qubit.index == node2.qubit.index:
                return (self.value_id, self.copy_id) > (node2.value_id, node2.copy_id)
            return self.qubit.index > node2.qubit.index
        return self.qubit.register.name > node2.qubit.register.name

    def qubitName(qubit: Qubit) -> str:
        return qubit.register.name + "_" + str(qubit.index)

class ValueIdNode:
    def __init__(self, valId: int):
        self.valId :int = valId
        self.edges: List[ValueIdEdge] = []
    def add_edge(self, edge: ValueIdEdge):
        self.edges.append(edge)
    
class ValueIdEdge:
    def __init__(self, gate: Instruction, ctrls: List[Tuple[str, int]], node_to: ValueIdNode, is_computation: bool):
        self.gate = gate
        self.ctrls = ctrls
        self.node_to = node_to
        self.is_computation = is_computation

    def checkCompatible(self, gate: Instruction, new_ctrls: List[Tuple[str, int]]):
        if gate.name != self.gate.name:
            return False
        old_ctrls = self.ctrls
        if len(old_ctrls) != len(new_ctrls):
            return False
        for (new_var_name, new_val_id) in new_ctrls:
            found = False
            for (old_var_name, old_val_id) in old_ctrls:
                if old_var_name == new_var_name and old_val_id == new_val_id:
                    found = True
                break
            if not found:
                return False
        return True

class ValueIdsGraph:
    def __init__(self):
        self.nodes: Dict[int, ValueIdNode] = {0: ValueIdNode(0)}
        self.first_free_val = 1

    def getValueIdAndUpdateGraph(self, previous_node: Node, ctrlNodes: List[Node], operation: Instruction, is_qfree: bool) -> int:
        ctrls_as_list = [(ctrl.variable_name, ctrl.value_id) for ctrl in ctrlNodes]
        if not is_qfree:
            node = ValueIdNode(self.first_free_val)
            self.first_free_val += 1
            self.nodes[node.valId] = node
            self.nodes[previous_node.value_id].add_edge(ValueIdEdge(operation, ctrls_as_list, node, True))
            return node.valId
        pred_node: ValueIdNode = self.nodes[previous_node.value_id]
        for edge in pred_node.edges:
            if edge.checkCompatible(operation, ctrls_as_list):
                return edge.node_to.valId
        # no known was found, we add a new edge and its reverse
        node = ValueIdNode(self.first_free_val)
        self.first_free_val += 1
        self.nodes[node.valId] = node
        self.nodes[previous_node.value_id].add_edge(ValueIdEdge(operation, ctrls_as_list, node, True))
        self.nodes[node.valId].add_edge(ValueIdEdge(operation, ctrls_as_list, self.nodes[previous_node.value_id], False))
        return node.valId

    def find_path(self, value_from: int, value_to: int) -> List[Tuple[int, bool]]: #list of (val id, is computation), does not contain value_from
        seen : Dict[int, bool] = {}
        queue: Deque[List[int]] = deque()
        queue.append([(value_from, True)])

        while queue:
            cur_path = queue.pop()
            (cur_node, is_comp) = cur_path[-1]
            if cur_node == value_to:
                return cur_path[1:]
            if seen.get(cur_node) is not None:
                continue
            seen[cur_node] = True
            for edge in self.nodes[cur_node].edges:
                cur_node_to = edge.node_to.valId
                new_path = deepcopy(cur_path)
                new_path.append((cur_node_to, cur_node < cur_node_to))
                queue.append(new_path)
        print("Could not find path from to in var compuation for " + str(value_from) + " to " + str(value_to))
        assert False
        
    def print(self):
        for (v, node) in self.nodes.items():
            for edge in node.edges:
                print("edge from " + str(node.valId) + " to " + str(edge.node_to.valId) + " with gate " + str(edge.gate))

    def is_linear(self) -> bool:
        #each node has either exactly one edge, or one forward and one backward
        def rec_aux(node: ValueIdNode):
            if len(node.edges) == 0:
                return True
            if len(node.edges) == 1:
                node_to = node.edges[0].node_to
                if node_to.valId < node.valId: #it s an uncomputation edge
                    return True
                return rec_aux(node_to)
            if len(node.edges) == 2:
                # we need to find the backward one
                node_to_0 = node.edges[0].node_to
                node_to_1 = node.edges[1].node_to
                if node_to_0.valId < node.valId: # this one is going back
                    if node_to_1.valId < node.valId: # two going back, should never happen
                        print("How did this happen: two back edges")
                        assert False
                    return rec_aux(node_to_1)
                if node_to_1.valId < node.valId: # this one is going back
                    return rec_aux(node_to_0)
                return False # two going forward,  not accepted
        return rec_aux(self.nodes[0])
        
    def find_pred(self, value):
        poss_parents = []
        for poss_parent in range(value):
            for edge in self.nodes[poss_parent].edges:
                if edge.node_to.valId == value:
                    poss_parents.append(poss_parent)
        assert len(poss_parents) == 1
        return poss_parents[0]

class CircuitGraph:
    def __init__(self, extra_qfree_gates: List[Instruction] = [], extra_non_qfree_gates: List[Instruction] = []):
        self.nodes: Dict[str, Dict[int, Node]] = {} #for a variable name, contains a list (ordered by value_id) of list (ordered by copy_id) of nodes
        self.extra_qfree_gates = extra_qfree_gates
        self.extra_non_qfree_gates = extra_non_qfree_gates
        self.value_id_graphs : Dict[str, ValueIdsGraph] = {}

    def buildFromOtherGraph(modelGraph: CircuitGraph): #create an empty circuit graph containing only init nodes for all non ancilla qubits.
        new_graph = CircuitGraph(modelGraph.extra_qfree_gates, modelGraph.extra_non_qfree_gates)
        for node_name in modelGraph.nodes:
            model_node = modelGraph.nodes[node_name][0][0]
            if not model_node.is_ancilla:
                node = Node(model_node.qubit, model_node.is_ancilla, model_node.is_qfree, model_node.gate, model_node.variable_name, model_node.value_id, model_node.copy_id)
                new_graph.addNode(node)
        return new_graph

    def addNode(self, node:Node) -> None:
        dict_nodes = self.nodes.get(node.variable_name)
        if dict_nodes is not None:
            if dict_nodes.get(node.value_id) is not None:
                assert len(dict_nodes[node.value_id]) == node.copy_id
                dict_nodes[node.value_id].append(node)
            else:
                assert node.copy_id == 0
                dict_nodes[node.value_id] = [node]
        else:
            assert node.value_id == 0 and node.copy_id == 0
            self.nodes[node.variable_name] = {node.value_id: [node]} 
            self.value_id_graphs[node.variable_name] = ValueIdsGraph()   

    def newCopyId(self, var_name: str, val_id: int) -> int:
        dict_nodes = self.nodes.get(var_name)
        if dict_nodes is None or dict_nodes.get(val_id) is None:
            return 0
        return len(dict_nodes[val_id])

    def getValueId(self, previous_node: Node, ctrlNodes: List[Node], operation: Instruction, is_qfree) -> int:
        return self.value_id_graphs[previous_node.variable_name].getValueIdAndUpdateGraph(previous_node, ctrlNodes, operation, is_qfree)

    def connectTargetNodes(self, node_from: Node, node_to: Node):
        assert node_to.target_edge_in is None
        assert node_from.target_edge_out is None
        e = Edge(node_from, node_to, 't')
        node_from.target_edge_out = e
        node_to.target_edge_in = e
        self._updatesAvailabilityEdges(e)
    
    def connectControlNodes(self, node_from:Node, node_to:Node, id_ctrl:int):
        assert node_from.variable_name != node_to.variable_name
        assert self.nodes[node_from.variable_name][node_from.value_id][node_from.copy_id] == node_from
        assert self.nodes[node_to.variable_name][node_to.value_id][node_to.copy_id] == node_to
        e = Edge(node_from, node_to, 'c', id_ctrl)
        node_from.edges_out.append(e)
        node_to.edges_in.append(e)
        self._updatesAvailabilityEdges(e)

    def _connectAvailabilityNodes(self, node_from:Node, node_to: Node):
        assert node_from.variable_name != node_to.variable_name
        e = Edge(node_from, node_to, 'a')
        node_from.edges_out.append(e)
        node_to.edges_in.append(e)

    def _updatesAvailabilityEdges(self, edge:Edge):
        new_edges: List[Edge] = []
        x1 = edge.node_from
        if edge.type == 't':
            #edge is some x1 -> x2, for all ctrl edges x1 o-> y, we add the availability one y --> x2
            x2 = edge.node_to
            for edge_out in x1.edges_out:
                if edge_out.type == 'c':
                    y = edge_out.node_to
                    e = self._connectAvailabilityNodes(y, x2)
        elif edge.type == 'c':
            #edge is some x1 -> y, if there is some target edge x1 -> x2, we add the availability one y -> x2
            y = edge.node_to
            if x1.target_edge_out is not None:
                x2 = x1.target_edge_out.node_to
                e = self._connectAvailabilityNodes(y, x2)

    def _aux_topoligical_sort(self, sorted_nodes: List[Node], seen_nodes: Dict[Node, int], node: Node):
        seen = seen_nodes.get(node)
        if seen is not None:
            if seen == 1:
                print("Cycle from node")
                print(node)
                assert False
            else:
                return
        seen_nodes[node] = 1 #to detect cycle, changed to 2 when the node is not active anymore
        for e in node.edges_out:
            self._aux_topoligical_sort(sorted_nodes, seen_nodes, e.node_to)
        if node.target_edge_out is not None:
            self._aux_topoligical_sort(sorted_nodes, seen_nodes, node.target_edge_out.node_to)
        seen_nodes[node] = 2
        sorted_nodes.append(node)

    def nodesInTopologicalOrder(self) -> List[Node]:
        sorted_nodes: List[Node] = []
        seen_nodes: Dict[Node, int] = {}
        for variable_name in self.nodes:
            for (val_id, list_copies) in self.nodes[variable_name].items():
                for node in list_copies:
                    self._aux_topoligical_sort(sorted_nodes, seen_nodes, node)
        sorted_nodes.reverse()
        return sorted_nodes

    def removeTargetEdge(self, edge: Edge):
        # first removes corresponding availability edges, then target edge
        assert edge.type == 't'
        # the edge is some x1 -> x2, so for all y such that x1 o-> y, we want to remove y --> x2 (note that because there is only one target edge going to x2, y --> x2 can only come from x1 -> x2)
        x1 = edge.node_from
        x2 = edge.node_to
        for edge in x1.edges_out:
            if edge.type == 'c':
                y = edge.node_to
                self._removeAvailEdge(y, x2)
        x1.target_edge_out = None
        x2.target_edge_in = None

    def removeCtrlEdge(self, edge: Edge):
        # first removes corresponding availability edges, then ctrl edge
        assert edge.type == 'c'
        # the edge is some x1 o-> y, so for the x2 such that x1 -> x2, we want to remove y --> x2 (note that because there is only one target edge going to x2, y --> x2 can only come from x1 -> x2)
        x1 = edge.node_from
        y = edge.node_to
        if x1.target_edge_out is not None:
            x2 = x1.target_edge_out.node_to
            self._removeAvailEdge(y, x2)
        x1.edges_out.remove(edge)
        y.edges_in.remove(edge)
    
    def _removeAvailEdge(self, y: Node, x2: Node):
        edge_removed = None
        for edge in y.edges_out:
            if edge.type == 'a' and edge.node_to == x2:
                assert edge_removed is None #there should only be one such edge
                edge_removed = edge
        assert edge_removed is not None
        y.edges_out.remove(edge_removed)
        x2.edges_in.remove(edge_removed)


    def latestCopy(self, node: Node) -> Node:
        assert self.nodes[node.variable_name] is not None and self.nodes[node.variable_name].get(node.value_id) is not None
        return self.nodes[node.variable_name][node.value_id][-1]

    def getHighestValueId(self, variable_name):
        max_vid = 0
        for k in self.nodes[variable_name].keys():
            max_vid = max(max_vid, k)
        return max_vid
    
    def graph_checks(self):
        for var_name in self.nodes:
            for v in self.nodes[var_name].keys():
                if v != 0:
                    for node in self.nodes[var_name][v]:
                        assert node.target_edge_in is not None
