from .computation_graph import ComputationGraph,compact_list_repr
import networkx as nx
import json

from typing import Union, Any, Callable
from graphviz import Digraph
from .computation_node import NodeContainer
from .computation_node import TensorNode, ModuleNode, FunctionNode
# from .utils import updated_dict, assert_input_type

COMPUTATION_NODES = Union[TensorNode, ModuleNode, FunctionNode]

class ComputationGraphV2(ComputationGraph):
    # def __init__(self):
    #     super(ComputationGraphV2,self).__init__()
    #     self.reset_nx_network()

    def __init__(
        self,
        visual_graph: Digraph,
        root_container: NodeContainer[TensorNode],
        show_shapes: bool = True,
        expand_nested: bool = False,
        hide_inner_tensors: bool = True,
        hide_module_functions: bool = True,
        roll: bool = True,
        depth: "int | float" = 3,
    ):
        super().__init__(
            visual_graph,
            root_container,
            show_shapes,
            expand_nested,
            hide_inner_tensors,
            hide_module_functions,
            roll,
            depth,
        )
        self.reset_nx_network()
        

    def reset_nx_network(self) -> None:
        '''
        Set template for networkx
        '''
        self.G: nx.Graph = None
        self.nodes_template = '{{ "id": "{}","name":"{}", "data": {{ "input_size": "{}", "output_size": "{}" }} }}'
        self.links_template = '{{ "source": "{}", "target": "{}", "weight": {} ,"count":{} }}'
        # self.network_template ='{{"nodes": [{}],"links": [{}]}}'
        self.network_template ='''
            {{
                "nodes": [{}],
                "links": [{}]
            }}
        '''
        self.networkx_nodes = []
        self.networkx_links = []

    def render_nx_network(self) -> nx.Graph:
        graph_data = self.network_template.format(
            ','.join(self.networkx_nodes),
            ','.join(self.networkx_links)
        )
        # print("Graph Data (Before JSON Load):", graph_data)
        graph_data_ = json.loads(graph_data)
        self.G = nx.node_link_graph(graph_data_)
        return self.G


    def get_node_for_nx(self, node: COMPUTATION_NODES) :# modified from ComputationGraph.get_node_label by kylin
        '''
        to be fill
        '''
        if self.show_shapes:
            if isinstance(node, TensorNode):
                node = self.nodes_template.format(self.id_dict[node.node_id],node.name,node.tensor_shape,node.tensor_shape)
            else:
                input_repr = compact_list_repr(node.input_shape)
                output_repr = compact_list_repr(node.output_shape)
                node = self.nodes_template.format(self.id_dict[node.node_id],node.name,input_repr,output_repr)
        return node


    def add_edge( # modified by kylin 
        # self, edge_ids: tuple[int, int], edg_cnt: int
        self, edge_ids, edg_cnt: int
    ) -> None:

        tail_id, head_id = edge_ids
        label = None if edg_cnt == 1 else f' x{edg_cnt}'

        # add by kylin
        nxlabel = f'{edg_cnt}'
        links = self.links_template.format(tail_id,head_id,'0',nxlabel)
        self.networkx_links.append(links)

        self.visual_graph.edge(f'{tail_id}', f'{head_id}', label=label)

    def add_node( # modified by kylin 
        # self, node: COMPUTATION_NODES, subgraph: Digraph | None = None
        self, node, subgraph
    ) -> None:
        '''Adds node to the graphviz with correct id, label and color
        settings. Updates state of running_node_id if node is not
        identified before.'''
        if node.node_id not in self.id_dict:
            self.id_dict[node.node_id] = self.running_node_id
            self.running_node_id += 1
        label = self.get_node_label(node)
        node_color = ComputationGraph.get_node_color(node)

        # add by kylin 
        nx_node = self.get_node_for_nx(node)
        self.networkx_nodes.append(nx_node)

        if subgraph is None:
            subgraph = self.visual_graph
        subgraph.node(
            name=f'{self.id_dict[node.node_id]}', label=label, fillcolor=node_color,
        ) # add node in visual_graph
        self.node_set.add(id(node))


        
