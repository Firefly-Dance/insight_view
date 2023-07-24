import torch
import timm

model_type = "GPUNet-0" # select one from above
precision = "fp32" # select either fp32 of fp16 (for better performance on GPU)
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_gpunet', pretrained=True,download = False, model_type=model_type, model_math=precision)


# # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

# model = timm.create_model("hf-hub:timm/eca_nfnet_l0", pretrained=True)
# model_type = "eca_nfnet_l0"


# input_size = [1]
# model = timm.create_model('inception_v4.tf_in1k', pretrained=True)
# size = model.pretrained_cfg['input_size']
# input_size.append(size[0,1,2] )

# model_graph_1 = draw_graph(
#         model, input_size=(1,3,224,224),
#         graph_name=model_type,
#         save_graph=True,
#         expand_nested = True
#     )

from torchview.torchviewV2 import draw_graph
#  import FunctionNode, ModuleNode, TensorNode, draw_graph
model_graph_2 = draw_graph(
        model, input_size=(1,3,224,224),
        graph_name=model_type+'2',
        save_graph=True,
        # expand_nested = True
    )

import networkx as nx

G = model_graph_2.render_nx_network()
print(G)

# def test():    
#     paths = []
#     path
#     if len(nextnode) == 1:
#         path,nextnode = init_linear_path(nextnode)

#     else:
#         path,nextnode = init_parallel_path(nextnode)

#     if bool(path):
#         paths.append(path)

# nodelist = []
# nextnode = list(model_graph_2.root_container)

# while len(nextnode):
#     node = nextnode.pop()
#     if isinstance(node,ModuleNode):
#         nodelist.append(node)

#     nextnode = list(node.children)


# print(model_graph_2.root_container)
# print(nodelist)

# from rebasin.permutation_initializer import PermutationInitializer
# permutation = PermutationInitializer(
#     model_a = model,
#     input_size_a = (1,size[0],size[1],size[2])
# )


# permutation = PermutationInitializer(
#     model_a = model,
#     input_size_a = (1,3,224,224)
# )

# path = permutation.initialize_permutations()

# print(path)


