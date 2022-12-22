from copy import deepcopy
from curses.ascii import ctrl
from itertools import count
from platform import node
from queue import PriorityQueue
from reqomp.circuit_graph import CircuitGraph, Node, Edge, InitGate, DeallocateGate
from reqomp.ancilla_order import AncillaOrderFromCircuitGraph
from qiskit.circuit import AncillaRegister # type: ignore
from typing import Dict, Tuple, Set, List
from collections import deque

from reqomp.dep_drawer import CircuitGraphDrawer

class NotEnoughAncillas(Exception):
    pass

def mark_can_reach_nodes(node: Node, circ_g: CircuitGraph, can_reach_nodes: Set[Node]): # mark all nodes that can reach orig node
    to_visit: deque[Node] = deque()
    to_visit.append(node)

    while to_visit:
        cur_n = to_visit.popleft()
        if cur_n in can_reach_nodes:
            continue
        can_reach_nodes.add(cur_n)
        if cur_n.target_edge_in is not None:
            to_visit.append(cur_n.target_edge_in.node_from)
        for edge in cur_n.edges_in:
            to_visit.append(edge.node_from) 

def mark_reachable_nodes(node: Node, circ_g: CircuitGraph, can_reach_nodes: Set[Node]): # mark all nodes that orig node can reach
    to_visit: deque[Node] = deque()
    to_visit.append(node)

    while to_visit:
        cur_n = to_visit.popleft()
        if cur_n in can_reach_nodes:
            continue
        can_reach_nodes.add(cur_n)
        if cur_n.target_edge_out is not None:
            to_visit.append(cur_n.target_edge_out.node_to)
        for edge in cur_n.edges_out:
            to_visit.append(edge.node_to) 

def is_available(node: Node, can_reach_node: Set[Node], reachable_nodes: Set[Node]) -> bool:
    if node in reachable_nodes:
        return False
    if node.target_edge_out is None:
        return True
    targ_succ = node.target_edge_out.node_to
    if targ_succ not in can_reach_node:
        return True
    return False

def _one_step(circ_g_u: CircuitGraph, circ_g: CircuitGraph, prev_node: Node, next_node_original: Node, is_computation: bool, order_value_ctrls: Dict[str, int], active_variables: Set[str], use_latest_ctrl: bool = True) -> Node: # circuit graph, and node to (un)compute next of, is computation (else uncomputation), and values ordering controls
    if is_computation:
        assert circ_g.nodes[prev_node.variable_name].get(prev_node.value_id) is not None # such a node was already computed once
    else:
        assert prev_node.value_id > 0

    # create the new node
    value_id = next_node_original.value_id
    value_id_to_copy_from = next_node_original.value_id if is_computation else prev_node.value_id
    copy_id = len(circ_g_u.nodes[prev_node.variable_name][value_id]) if circ_g_u.nodes[prev_node.variable_name].get(value_id) is not None else 0
    first_comp_node = circ_g.nodes[prev_node.variable_name][value_id_to_copy_from][0]
    assert first_comp_node.is_qfree or (copy_id == 0 and not prev_node.is_ancilla)
    new_computed_node = Node(prev_node.qubit, prev_node.is_ancilla, first_comp_node.is_qfree, first_comp_node.gate, prev_node.variable_name, value_id, copy_id)
    circ_g_u.addNode(new_computed_node)

    # update target links
    if prev_node.target_edge_out is None:
        circ_g_u.connectTargetNodes(prev_node, new_computed_node)
    else:
        raise NotEnoughAncillas
    
    old_ctrls = [ctrl_edge for ctrl_edge in first_comp_node.edges_in if ctrl_edge.type == 'c']
    def get_ctrl_id(ctrl_edge: Edge):
        def get_max_val_on_wire(node: Node):
            #if multiple ancillas share the same wire, the order that matters is the biggest one see plr_bug 2
            val = order_value_ctrls[node.variable_name]
            #going up the target edges
            while node.target_edge_in is not None:
                node = node.target_edge_in.node_from
                val = max(val, order_value_ctrls[node.variable_name])
            #going down the target edges
            while node.target_edge_out is not None:
                node = node.target_edge_out.node_to
                val = max(val, order_value_ctrls[node.variable_name])
            return val
        return (ctrl_edge.node_from.is_ancilla, get_max_val_on_wire(ctrl_edge.node_from)) #i want ancilla after ctrls in non rev
    new_ctrls: List[Node] = sorted(old_ctrls, key = get_ctrl_id, reverse=True)

    for edge in new_ctrls:
        c = edge.node_from
        if circ_g_u.nodes[c.variable_name].get(c.value_id) is not None:
            (can_reach_node, reachable_nodes) = (set(), set())
            mark_can_reach_nodes(new_computed_node, circ_g_u, can_reach_node)
            mark_reachable_nodes(new_computed_node, circ_g_u, reachable_nodes)
            if use_latest_ctrl:
                latest_copy = circ_g_u.nodes[c.variable_name][c.value_id][-1]
                if is_available(latest_copy, can_reach_node, reachable_nodes):
                    circ_g_u.connectControlNodes(latest_copy, new_computed_node, edge.id_ctrl)
                    continue
            else:
                found = False
                for cur_copy in circ_g_u.nodes[c.variable_name][c.value_id]:
                    if is_available(cur_copy, can_reach_node, reachable_nodes):
                        circ_g_u.connectControlNodes(cur_copy, new_computed_node, edge.id_ctrl)
                        found = True
                        break
                if found:
                    continue
        latest_value: Node = circ_g_u.nodes[c.variable_name][circ_g_u.getHighestValueId(c.variable_name)][-1]
        while latest_value.target_edge_out is not None:
            latest_value = latest_value.target_edge_out.node_to
        latest_value = _multistep(circ_g_u, circ_g, latest_value, c.value_id, order_value_ctrls, active_variables, use_latest_ctrl=use_latest_ctrl)
        (can_reach_node, reachable_nodes) = (set(), set())
        mark_can_reach_nodes(new_computed_node, circ_g_u, can_reach_node)
        mark_reachable_nodes(new_computed_node, circ_g_u, reachable_nodes)
        if not is_available(latest_value, can_reach_node, reachable_nodes):
            raise NotEnoughAncillas
        circ_g_u.connectControlNodes(latest_value, new_computed_node, edge.id_ctrl)
    return new_computed_node

def _multistep(circ_g_u: CircuitGraph, circ_g: CircuitGraph, node_from: Node, value_id_obj: int, order_value_ctrls: Dict[str, int], active_variables: Set[str], use_latest_ctrl: bool = True) -> Node:
    assert node_from.variable_name not in active_variables
    active_variables.add(node_from.variable_name)
    prev_node = node_from
    val_ids_list = circ_g.value_id_graphs[prev_node.variable_name].find_path(node_from.value_id, value_id_obj)

    # if some control has a back and forth (in grover: p0 -> p1 et p0->p2, and we compute p2 before, then p1 is never reachable again => in that case we want to force the computation of p1 then that of p2)
    #this is only a problem for non ancillas, as ancillas are qfree and back and forth are possible
    if not node_from.is_ancilla:
        vals_range = range(node_from.value_id + 1, value_id_obj + 1) if node_from.value_id < value_id_obj else range(value_id_obj, node_from.value_id)
        missing_values = [i for i in vals_range if (i, True) not in val_ids_list and (i, False) not in val_ids_list]
        if len(missing_values) > 0:
            #print("There were missing values on " + str(node_from) + " : " + str(missing_values))
            missing_values.sort()
        new_path = []
        prev_val = node_from.value_id
        for v in missing_values:
            new_path += circ_g.value_id_graphs[prev_node.variable_name].find_path(prev_val, v)
            prev_val = v
        new_path += circ_g.value_id_graphs[prev_node.variable_name].find_path(prev_val, value_id_obj)
        val_ids_list = new_path

    for (val_id, is_comp) in val_ids_list:
        next_node_original = circ_g.nodes[prev_node.variable_name][val_id][0]
        prev_node = _one_step(circ_g_u, circ_g, prev_node, next_node_original, is_computation=is_comp, order_value_ctrls = order_value_ctrls, active_variables= active_variables, use_latest_ctrl= use_latest_ctrl)
    active_variables.remove(node_from.variable_name)
    return prev_node

def lazy_uncomputation(circ_g: CircuitGraph, ordered_ancilla_deps: List[List[str]]):
    latest_val_id: Dict[str, int] = {}
    for node_name in circ_g.nodes:
        if circ_g.nodes[node_name][0][0].is_ancilla:
            node = circ_g.nodes[node_name][circ_g.getHighestValueId(node_name)][-1]
            while node.target_edge_out is not None:
                node = node.target_edge_out.node_to
            latest_val_id[node.variable_name] = node.value_id

    nodes_to_uncomp = [n for n in circ_g.nodesInTopologicalOrder() if n.is_ancilla and n.value_id > 0 and n.value_id <= latest_val_id[n.variable_name] and n.copy_id == 0]
    nodes_to_uncomp.reverse()
    ctrls_order = AncillaOrderFromCircuitGraph(circ_g)
    ordered_values_controls = ctrls_order.get_order_valued_controls()
    active_variables: Set[str] = set()
    circ_g_u: CircuitGraph = deepcopy(circ_g)
    for node in nodes_to_uncomp:
        last_node_on_qb = circ_g_u.nodes[node.variable_name][node.value_id][-1]
        assert last_node_on_qb.target_edge_out is None
        pred_val = circ_g.value_id_graphs[node.variable_name].find_pred(node.value_id)
        assert pred_val == node.value_id - 1
        uncomp_node = _multistep(circ_g_u, circ_g, last_node_on_qb, pred_val, ordered_values_controls, active_variables, use_latest_ctrl=False)

    checks_last_val_on_qb(circ_g, circ_g_u)
    return reuse_ancilla_registers(circ_g_u, ordered_ancilla_deps)

def checks_last_val_on_qb(circ_g: CircuitGraph, circ_g_u: CircuitGraph):
    def gets_last_node_on_wire(node: Node) -> Node:
        while node.target_edge_out is not None:
            node = node.target_edge_out.node_to
        return node

    for var in circ_g.nodes:
        last_node_g: Node = gets_last_node_on_wire(circ_g.nodes[var][0][0])
        last_node_g_u: Node = gets_last_node_on_wire(circ_g_u.nodes[var][0][0])
        if last_node_g.is_ancilla and last_node_g_u.value_id != 0:
            print("last state ancilla was wrong")
            assert False
        elif not last_node_g.is_ancilla and last_node_g.value_id != last_node_g_u.value_id:
            print("last state control was wrong")
            assert last_node_g.value_id == last_node_g_u.value_id

def reuse_ancilla_registers(circ_g: CircuitGraph, ordered_ancilla_deps: List[List[str]]):
    # insert deallocation nodes (to avoid linking a_1 to a_2 if there is a dependency edge between the 2)
    for var in circ_g.nodes:
        dict_nodes = circ_g.nodes[var]
        if dict_nodes is not None and dict_nodes.get(0) is not None:
            cur_node = dict_nodes[0][-1]
            if cur_node.is_ancilla:
                if cur_node.target_edge_out is None:
                    de_node = Node(cur_node.qubit, is_anc=True, is_qfr=True, gate=DeallocateGate(), var_name=cur_node.variable_name, value_id=0, copy_id=len(circ_g.nodes[cur_node.variable_name][0]))
                    circ_g.addNode(de_node)
                    circ_g.connectTargetNodes(cur_node, de_node)
                else:
                    print("Last vertex on ancilla has target edge out:" + str(cur_node))
                    assert False
        else:
            print("A variable with no nodes " + var)
            assert False

    #makes ordering
    counter_val = {}
    cur_v = 0
    if ordered_ancilla_deps is not None:
        for component in ordered_ancilla_deps:
            for ancilla in component:
                counter_val[ancilla] = cur_v
                cur_v += 1
    else:
        # we just give the values in alphabetic order, hoping that reflects order of creation of ancillas
        for node in sorted(circ_g.nodes):
            if circ_g.nodes[node][0][0].is_ancilla:
                counter_val[node] = cur_v
                cur_v += 1
    

    # we now go through the nodes in topo order with extra constraint for equality: avoid taking ancilla first nodes as much as possible, and when we have to, pick the ones with oldest date of birth
    #when we get an ancilla first node, if possible we link it to another ancilla dealloc node, and change the register of the second to that of the first
    not_ancilla_alloc: Node = [] # parentless nodes that are not an ancilla alloc, simple list
    ancilla_alloc: PriorityQueue[Node] = PriorityQueue() #when pop: element with least prio comes
    for var in circ_g.nodes:
        # no need to look into the deallocate nodes, they all have a parent for now
        dict_nodes: Dict[int, List[Node]] = circ_g.nodes[var]
        if dict_nodes is not None and dict_nodes.get(0) is not None:
            node: Node = dict_nodes[0][0]
            if node.target_edge_in is None and not node.edges_in:
                assert isinstance(node.gate, InitGate) 
                if node.is_ancilla:
                    ancilla_alloc.put((counter_val[node.variable_name], node)) 
                else:
                    not_ancilla_alloc.append(node)
            else:
                assert not isinstance(node.gate, InitGate) 

    available_ancilla_slots: List[Node] = [] # ancillas that have already been deallocated, hence that new ancilla alloc can be linked to
    removed_edges = {} # for each node: remove_edges[node] = [nb_removed_target_edges, nb_removed_ctrl_edges, nb_removed_non_Ctrl_edges]
    const_target = 0
    const_nontarg = 1
    # because we can't remove them really because they're needed to build the dag
    while not_ancilla_alloc or not ancilla_alloc.empty():
        cur_node = None
        # get the node, and deal with ancilla spots
        if not_ancilla_alloc:
            cur_node: Node = not_ancilla_alloc.pop()
            if cur_node.gate.name == 'deallocate':
                available_ancilla_slots.append(cur_node)
        else:
            (allocation_date, cur_node) = ancilla_alloc.get()
            if available_ancilla_slots:
                # we link the two ancillas: add a target link and change register of the new ancilla to that of the old one
                old_node = available_ancilla_slots.pop()
                circ_g.connectTargetNodes(old_node, cur_node)
                node_corrected = cur_node
                while(node_corrected.target_edge_out is not None):
                    node_corrected.qubit = old_node.qubit
                    node_corrected = node_corrected.target_edge_out.node_to
                node_corrected.qubit = old_node.qubit
            else: #we create a new single qubit, otherwise we may have a big regitser of many qubits whereas we only need one
                anc_reg = AncillaRegister(1, name = "anc" + Node.qubitName(cur_node.qubit))
                node_corrected = cur_node
                while(node_corrected.target_edge_out is not None):
                    node_corrected.qubit = anc_reg[0]
                    node_corrected = node_corrected.target_edge_out.node_to
                node_corrected.qubit = anc_reg[0]
            
        # remove all outgoing edges from this node form dep_graph, and add new orphan nodes to lists
        if cur_node.target_edge_out is not None:
            target_son: Node = cur_node.target_edge_out.node_to
            if removed_edges.get(target_son) is None:
                removed_edges[target_son] = [0, 0]
            removed_edges[target_son][const_target] += 1
            if removed_edges[target_son][const_nontarg] == len(target_son.edges_in):
                not_ancilla_alloc.append(target_son) # it consumes some node, so not ancilla alloc
        for e in cur_node.edges_out:
            node_to = e.node_to
            if removed_edges.get(node_to) is None:
                removed_edges[node_to] = [0, 0]
            removed_edges[node_to][const_nontarg] += 1
            if removed_edges[node_to][const_nontarg] == len(node_to.edges_in):
                if removed_edges[node_to][const_target] == 1:
                    not_ancilla_alloc.append(node_to) # it consumes some node, so not ancilla alloc
                elif node_to.target_edge_in is None:
                    if node_to.is_ancilla:
                        ancilla_alloc.put((counter_val[node_to.variable_name], node_to))
                    else:
                        not_ancilla_alloc.append(node_to)
                        
    return circ_g
    

def _linear_hierarchy_uncomputation_strategy(nb_ancillas: int, nb_qubits: int, memo_table: Dict[Tuple[int ,int, bool], Tuple[int, List[Tuple[int, bool]]]], clean_up_last = False): # we store (nbStepsRequired, List [(idAncilla, is_compute)]) (if not is_compute: it is an uncomputation step)
    if nb_ancillas == 1 and nb_qubits >=1:
        if clean_up_last:
            return (2, [(0, True), (0, False)])
        else:
            return (1, [(0, True)])
    if nb_ancillas >= 2 and nb_qubits == 1:
        return (-1, [])
    if memo_table.get((nb_ancillas, nb_qubits, clean_up_last)) is not None:
        return memo_table.get((nb_ancillas, nb_qubits, clean_up_last))
    min_nb_steps = -1
    best_moves = []
    for m in range(nb_ancillas - 1, 0, -1):
        if not clean_up_last:
            (nb_ops1, l1) = _linear_hierarchy_uncomputation_strategy(m, nb_qubits, memo_table, clean_up_last=False)
            (nb_ops2, l2) = _linear_hierarchy_uncomputation_strategy(nb_ancillas - m, nb_qubits - 1, memo_table, clean_up_last=False)
            (nb_ops3, l3) = _linear_hierarchy_uncomputation_strategy(m, nb_qubits - 1, memo_table, clean_up_last=False)
            if(nb_ops1 == -1 or nb_ops2 == -1 or nb_ops3 == -1):
                continue
            cur_nb_steps = nb_ops1 + nb_ops2 + nb_ops3
            if(min_nb_steps == -1 or cur_nb_steps < min_nb_steps):
                best_moves = [a for a in l1] # first we compute the m'th ancilla, and clean up all the others
                updated_l2 = [(a + m, b) for (a, b) in l2] # then we compute the n'th ancilla from the m'th one, and clean up
                best_moves += updated_l2
                updated_l3 = [(a, not b) for (a, b) in l3]
                updated_l3.reverse()
                best_moves += updated_l3
                min_nb_steps = cur_nb_steps
        else:
            (nb_ops1, l1) = _linear_hierarchy_uncomputation_strategy(m, nb_qubits, memo_table, clean_up_last=False)
            (nb_ops2, l2) = _linear_hierarchy_uncomputation_strategy(nb_ancillas - m, nb_qubits - 1, memo_table, clean_up_last=True)
            (nb_ops3, l3) = _linear_hierarchy_uncomputation_strategy(m, nb_qubits, memo_table, clean_up_last=False) # we uncompute the first qubit straight away -> we still have n-1 qubits available
            if(nb_ops1 == -1 or nb_ops2 == -1 or nb_ops3 == -1):
                continue
            cur_nb_steps = nb_ops1 + nb_ops2 + nb_ops3
            if(min_nb_steps == -1 or cur_nb_steps < min_nb_steps):
                best_moves = [a for a in l1] # first we compute the m'th ancilla, and clean up all the others
                updated_l2 = [(a + m, b) for (a, b) in l2] # then we compute the n'th ancilla from the m'th one, and clean up
                best_moves += updated_l2
                updated_l3 = [(a, not b) for (a, b) in l3]
                updated_l3.reverse()
                best_moves += updated_l3
                min_nb_steps = cur_nb_steps
    memo_table[(nb_ancillas, nb_qubits, clean_up_last)] = (min_nb_steps, best_moves)
    return (min_nb_steps, best_moves)

def _check_uncomputation_strategy(nb_ancillas: int, nb_qubits: int, steps: List[Tuple[int, bool]]):
    nb_free = nb_qubits
    computed = [False for i in range(nb_ancillas)]

    for (anc, is_comp) in steps:
        if is_comp:
            assert not computed[anc]
            computed[anc] = True
            assert nb_free > 0
            nb_free -= 1
        else:
            assert computed[anc]
            computed[anc] = False
            assert nb_free < nb_qubits
            nb_free += 1

def hierarchical_uncomputation(circ_g: CircuitGraph, nb_qubits: int, force_lazy = False) -> CircuitGraph:
    ancilla_order = AncillaOrderFromCircuitGraph(circ_g)
    value_order_ctrls = ancilla_order.get_order_valued_controls()
    try:
        ordered_ancilla_deps: List[List[str]] = ancilla_order.get_ordered_ancilla_dependency_graphs()
    except AssertionError: # ancilla dependency has a cycle
        print("WARNING: Ancillas dependencies too complex for hierarchical uncomputation, falling back to lazy.")
        return lazy_uncomputation(circ_g, None)
    for connected_component in ordered_ancilla_deps:
        if not ancilla_order.anc_dep_g.is_linear(connected_component):
            print("WARNING: the following ancillas did not have a linear dependency, falling back to lazy.")
            for n in ancilla_order.anc_dep_g.nodes:
                print(n + ", ", end = "")
            print(" ")
            return lazy_uncomputation(circ_g, ordered_ancilla_deps)
    if force_lazy:
        return lazy_uncomputation(circ_g, ordered_ancilla_deps)
    #check that all qubits value id graphs are a line, if this is not the case, fallback to lazy.
    for var_name in circ_g.nodes:
        if circ_g.nodes[var_name][0][0].is_ancilla and not circ_g.value_id_graphs[var_name].is_linear():
            print("WARNING: Value Ids graph of qubit " + var_name + " is not linear, falling back to lazy.")
            return lazy_uncomputation(circ_g, ordered_ancilla_deps)
    #ordered_ancilla_deps: List[List[str]] = ancilla_order.get_ordered_ancilla_dependency_graphs()
    linear_uncomputations: Dict[Tuple[int ,int], Tuple[int, List[Tuple[int, bool]]]] = {}
    available_ancillas: Set[AncillaRegister] = {AncillaRegister(1) for i in range(nb_qubits)}
    last_on_qubit: Dict[AncillaRegister, Node] = {}
    computed_ancillas: Set[str] = set()
    circ_g_u = CircuitGraph.buildFromOtherGraph(circ_g)

    def get_last_on_qb(register: AncillaRegister):
        #it may be changed by some hidden recomputation, so we always follow down target edges
        cur_n = last_on_qubit.get(register)
        if cur_n is None:
            return None
        while cur_n.target_edge_out is not None:
            cur_n = cur_n.target_edge_out.node_to
        last_on_qubit[register] = cur_n
        return cur_n

    for connected_component in ordered_ancilla_deps:
        (nbSteps, steps) = _linear_hierarchy_uncomputation_strategy(len(connected_component), nb_qubits, linear_uncomputations, clean_up_last = True)
        if nbSteps == -1:
            print("ERROR: Not enough qubits for " + str(len(connected_component)) + "ancillae on " + str(nb_qubits) + "qubits .")#, changed to " + str(nb_qubits))
            raise NotEnoughAncillas
        
        pos_last_comp = [-1 for t in connected_component]
        for i in range(len(steps)):
            (a, is_comp) = steps[i]
            if not is_comp:
                continue
            pos_last_comp[a] = i


        for i in range(nbSteps):
            (id_ancilla, is_computation) = steps[i]
            ancilla = connected_component[id_ancilla]
            if is_computation:
                register = None
                if len(available_ancillas) > 0:
                    register = available_ancillas.pop()
                else:
                    print("Linear Strategy is broken.")
                    raise NotEnoughAncillas
                previous_computations = circ_g_u.nodes.get(ancilla)
                copy_id = 0
                if previous_computations is not None:
                    copy_id = len(circ_g_u.nodes[ancilla][0])
                new_node = Node(qubit = register[0], is_anc = True, is_qfr = True, gate = InitGate(), value_id = 0, copy_id = copy_id, var_name = ancilla)
                circ_g_u.addNode(new_node)
                if get_last_on_qb(register) is not None:
                    circ_g.connectTargetNodes(get_last_on_qb(register), new_node)
                last_comp_node = _multistep(circ_g_u, circ_g, new_node, circ_g.getHighestValueId(ancilla), value_order_ctrls, set(), use_latest_ctrl=True)
                last_on_qubit[register] = last_comp_node
                if pos_last_comp[id_ancilla] == i:
                    for (k, list_copies) in circ_g.nodes[ancilla].items():
                        for node in list_copies:
                            for ctrl_edge in node.edges_out:
                                if ctrl_edge.type == 'c' and not ctrl_edge.node_to.is_ancilla:
                                    node_to = ctrl_edge.node_to
                                    if circ_g_u.nodes[node_to.variable_name].get(node_to.value_id) is None:
                                        old_node = circ_g_u.nodes[node_to.variable_name][circ_g_u.getHighestValueId(node_to.variable_name)][-1]
                                        while old_node.target_edge_out is not None:
                                            old_node = old_node.target_edge_out.node_to
                                        last_comp_node = _multistep(circ_g_u, circ_g, old_node, node_to.value_id, value_order_ctrls, set(), use_latest_ctrl=True)

            else: #this is uncomputation
                last_comp_node = circ_g_u.nodes[ancilla][circ_g_u.getHighestValueId(ancilla)][-1]
                #but there may be uncomputation already from the circuit itself in here, so we should follow targte edges out
                # TODO could be dangerous if linked to another variable -> but no, bc uncomputation, so there can't be anything linked, the qb is not free
                while(last_comp_node.target_edge_out is not None):
                    last_comp_node = last_comp_node.target_edge_out.node_to
                assert get_last_on_qb(last_comp_node.qubit.register) == last_comp_node
                uncomputed_node = _multistep(circ_g_u, circ_g, last_comp_node, 0, value_order_ctrls, set(), use_latest_ctrl=True)
                last_on_qubit[uncomputed_node.qubit.register] = uncomputed_node
                available_ancillas.add(last_comp_node.qubit.register)

    #make sure all outputs are computed
    for variable_name in circ_g.nodes:
        if not circ_g.nodes[variable_name][0][0].is_ancilla:
            cur_last_node = circ_g_u.nodes[variable_name][circ_g_u.getHighestValueId(variable_name)][-1]
            while cur_last_node.target_edge_out is not None:
                cur_last_node = cur_last_node.target_edge_out.node_to
            obj_last_node = circ_g.nodes[variable_name][circ_g.getHighestValueId(variable_name)][-1]
            while obj_last_node.target_edge_out is not None:
                obj_last_node = obj_last_node.target_edge_out.node_to
            updated_node = _multistep(circ_g_u, circ_g, cur_last_node, obj_last_node.value_id, value_order_ctrls, set(), use_latest_ctrl=True)
    
    checks_last_val_on_qb(circ_g, circ_g_u)

    return circ_g_u
    
