from __future__ import annotations
from reqomp.circuit_graph import CircuitGraph, Node
from typing import Dict, Optional, Set, List, Tuple
#from graphviz import Digraph # type: ignore
# to use the DependencyGraphDrawer, uncomment the above


class DependencyGraphDrawer:
    def nodeName(self, node):
        return node.variable_name

    def draw(self, dep_graph: DependencyGraph, filename = 'graph.gvdot'):
        g = Digraph('DependencyGraph', filename=filename)
        
        #create all the nodes
        for var_name in dep_graph.nodes:
            g.node(var_name)
        
        #add all the edges-> we only take care of outgoing
        for var_name, node in dep_graph.nodes.items():
            for node_to in node.nodes_out:
                g.edge(self.nodeName(node), self.nodeName(node_to))
        g.save()
        return g

class DependencyNode:
    def __init__(self, var_name: str):
        self.variable_name = var_name
        self.nodes_in: List[DependencyNode] = []
        self.nodes_out: List[DependencyNode] = []

    def __repr__(self) -> str:
        return "DependancyNode: var({})".format(self.variable_name,)

    def __gt__(self, node2: DependencyNode) -> bool: #order does not matter so much, we just want to break ties when comparing ancillaRegisters
        return self.variable_name > node2.variable_name

class DependencyGraph:
    def __init__(self):
        self.nodes: Dict[str, DependencyNode] = {} #for a given variable name, contains the corresponding node

    def add_edge(self, var_from: str, var_to: str):
        node_from = self.nodes[var_from]
        node_to = self.nodes[var_to]
        node_from.nodes_out.append(node_to)
        node_to.nodes_in.append(node_from)
    
    def add_node(self, var_name: str):
        assert self.nodes.get(var_name) is None
        node = DependencyNode(var_name)
        self.nodes[var_name] = node

    def aux_dfs_topo_sort(self, node, seen, ordered_nodes, on_cycle_fail:bool, parent_node: DependencyNode):
        s = seen.get(node)
        if s == 1:
            return
        if s == 2:
            if on_cycle_fail:
                assert False
            # we simply remove the offending edge: we want some reasonable ordering of qubits, following as closely as possible their dependencies, no more
            parent_node.nodes_out.remove(node)
            node.nodes_in.remove(parent_node)
            seen[node] = 1
            return
        seen[node] = 2
        for next in node.nodes_out:
            self.aux_dfs_topo_sort(next, seen, ordered_nodes, on_cycle_fail, node)
        ordered_nodes.append(node)
        seen[node] = 1

    def nodes_in_topological_order(self, on_cycle_fail: bool):
        seen: Dict[DependencyNode, int] = {}
        ordered_nodes: List[DependencyNode] = []
        
        for v_n, node in self.nodes.items():
            self.aux_dfs_topo_sort(node, seen, ordered_nodes, on_cycle_fail, node)
        ordered_nodes.reverse()
        return ordered_nodes

    def is_linear(self, connected_comp: List[str]) -> bool:
        for id_n in connected_comp:
            node = self.nodes[id_n]
            if len(node.nodes_in) > 1 or len(node.nodes_out) > 1:
                return False
        return True

class AncillaOrderFromCircuitGraph:
    def __init__(self, circ_graph: CircuitGraph):
        self.circ_g = circ_graph
        self.anc_dep_g = DependencyGraph()
        self.ctrls_dep_g = DependencyGraph()
        self.ctrls_order_values: Dict[str, int] = {} # an int value for each non ancilla qbit, such that controls should be ordered as their values are
        self.connected_components: Dict[str, int] = {}
        self.nb_connected_components = 0
        self.reachable_connected_components: Dict[Node, List[int]] = {} # for each (circuit graph) node, reachable_connected_components[node][i] = 0/1/2 if the i'th connected component cannot be reached/can be reached only in a qfree manner/can be reached in a non qfree manner
        self.cc_graph = DependencyGraph()
        self.ordered_ccs :List[int] = []


    def _build_dependency_graph(self, dep_g: DependencyGraph, only_ancillas: bool):
        for variable in self.circ_g.nodes:
            if not only_ancillas or self.circ_g.nodes[variable][0][0].is_ancilla:
                dep_g.add_node(variable)
        for variable in self.circ_g.nodes:
            if not only_ancillas or self.circ_g.nodes[variable][0][0].is_ancilla:
                # we check all nodes on this ancilla for control edges to another ancilla
                cur_node: Node = self.circ_g.nodes[variable][0][0]
                while True:
                    for edge in cur_node.edges_out:
                        if edge.type == 'c' and (not only_ancillas or edge.node_to.is_ancilla):
                            dep_g.add_edge(cur_node.variable_name, edge.node_to.variable_name)
                    if cur_node.target_edge_out is not None:
                        cur_node = cur_node.target_edge_out.node_to
                    else:
                        break
        return dep_g
    
    def _mark_connected_component(self, node: DependencyNode, id_CC: int):
        cur_cc = self.connected_components.get(node.variable_name)
        if cur_cc is not None:
            assert cur_cc == id_CC
            return
        self.connected_components[node.variable_name] = id_CC
        for node_to in node.nodes_out:
            self._mark_connected_component(node_to, id_CC)
        for node_from in node.nodes_in:
            self._mark_connected_component(node_from, id_CC)
        
    def _identify_connected_components(self):
        id_connected_comp = 0
        for var_name, node in self.anc_dep_g.nodes.items():
            if self.connected_components.get(var_name) is None:
                self._mark_connected_component(node, id_connected_comp)
                id_connected_comp += 1
        return id_connected_comp

    def _get_reachables_ccs(self, node: Node):
        res = self.reachable_connected_components.get(node)
        if res is not None:
            return res
        res = [0 for i in range(self.nb_connected_components)]
        is_non_qfree_coeff = 1
        if node.is_ancilla:
            res[self.connected_components[node.variable_name]] = 1
        elif not node.is_qfree:
            is_non_qfree_coeff = 2
        for edge in node.edges_out:
                t = self._get_reachables_ccs(edge.node_to)
                for i in range(len(res)):
                    res[i] = max(res[i], min(is_non_qfree_coeff * t[i], 2)) # if cur gate is non qfree, we want 2 -> 2, 1 -> 2, 0 -> 0
        if node.target_edge_out is not None:
            t = self._get_reachables_ccs(node.target_edge_out.node_to)
            for i in range(len(res)):
                    res[i] = max(res[i], min(is_non_qfree_coeff * t[i], 2))
        # we are using this ordering to link ancillas, so we need to take care of the additional availability edges that would be created by linking.
        # so we could add dummy end wire nodes to every ancilla. To avoid extra work on the graph, we simply add the links those dummy nodes would create:
        # for any node c such that an ancilla node a controls it: a o-> c, if a is the last node of the ancilla, the dummy node would simply add that a 's CC is reachable from c. And even if a is not the last on the node, we already have that a 's cc is reachable from c (because of a's next node).
        # so altogether: for any input control edge a o-> c, we add a link to a 's cc from c
        for edge in node.edges_in:
            node_from = edge.node_from
            if edge.type == 'c' and node_from.is_ancilla:
                res[self.connected_components[node_from.variable_name]] = max(1, self.connected_components[node_from.variable_name])
        self.reachable_connected_components[node] = res
        return res

    def _order_connected_components(self):
        for id_c in range(self.nb_connected_components):
            self.cc_graph.add_node(str(id_c))
        for ancilla_name in self.anc_dep_g.nodes:
            reachable_ccs = self._get_reachables_ccs(self.circ_g.nodes[ancilla_name][0][0])
            for cur_cc in range(len(reachable_ccs)):
                if cur_cc != self.connected_components[ancilla_name] and reachable_ccs[cur_cc] > 0:
                    self.cc_graph.add_edge(str(self.connected_components[ancilla_name]), str(cur_cc))
        
        ordered_ccs_as_nodes = self.cc_graph.nodes_in_topological_order(on_cycle_fail=True)
        self.ordered_ccs = [int(node.variable_name) for node in ordered_ccs_as_nodes]

    def get_ordered_ancilla_dependency_graphs(self) -> List[List[str]]: 
        # each list of nodes is a dependency graph connected component
        self._build_dependency_graph(self.anc_dep_g, True)
        self.nb_connected_components = self._identify_connected_components()
        self._order_connected_components()
        ordered_anc = self.anc_dep_g.nodes_in_topological_order(on_cycle_fail=True) # TODO fail in a nicer way
        res: List[List[str]] = []
        for id_c in self.ordered_ccs:
            cur_g = [node.variable_name for node in ordered_anc if self.connected_components[node.variable_name] == id_c]
            res.append(cur_g)
        return res

    def get_order_valued_controls(self) -> Dict[str, int]:
        self._build_dependency_graph(self.ctrls_dep_g, False)
        ordered_ctrls: List[DependencyNode] = self.ctrls_dep_g.nodes_in_topological_order(on_cycle_fail=False)
        val_ctrl = 0
        for ctrl in ordered_ctrls:
            self.ctrls_order_values[ctrl.variable_name] = val_ctrl
            val_ctrl +=1
        return self.ctrls_order_values
