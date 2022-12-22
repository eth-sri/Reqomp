#from graphviz import Digraph # type: ignore
# to use the CircuitGraphDrawer, uncomment the above

from reqomp.circuit_graph import CircuitGraph, Node, Edge

class CircuitGraphDrawer:
    def nodeName(self, node):
        return node.variable_name + "_" + str(node.value_id) + "_" + str(node.copy_id) + "\n" + str(node.gate.name)

    def draw(self, circuit_graph, filename = 'graph.dot'):
        g = Digraph('CircuitGraph', filename=filename)
        
        #create all the nodes
        for variable_name in circuit_graph.nodes:
            for (k, list_copies) in circuit_graph.nodes[variable_name].items():
                for node in list_copies:
                    #print(node)
                    g.node(self.nodeName(node), group = node.variable_name)
        
        #add all the edges-> we only take care of incoming
        for variable_name in circuit_graph.nodes:
            for (k, list_copies) in circuit_graph.nodes[variable_name].items():
                for node in list_copies:
                    for edge in node.edges_in:
                        g.edge(self.nodeName(edge.node_from), self.nodeName(node), edge.type)
                    if node.target_edge_in is not None:
                        g.edge(self.nodeName(node.target_edge_in.node_from), self.nodeName(node), label='t')
        g.save()
        return g
  
