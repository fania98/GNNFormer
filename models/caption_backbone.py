# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from .utils import NestedTensor, is_main_process

from .position_encoding import build_position_encoding, PositionEmbeddingSineGraphNode
from models.GNNEmbedding.gin_embedding import GIN
from models.GNNEmbedding.patch_feature_extractor import PatchFeatureExtractor
from models.swin_transformer import SwinTransformer
import dgl
import torchvision.models as tvmodels

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class SwinTransformerBackBone(nn.Module):
    def __init__(self, config):
        super(SwinTransformerBackBone, self).__init__()
        self.config = config
        self.swin = SwinTransformer(img_size=config.img_size, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=config.swin_embed_dim, depths=config.swin_depth, num_heads=config.swin_num_heads,
                 window_size=config.swin_window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False)
        self.num_channels = config.swin_embed_dim*8
    def forward(self, tensor_list: NestedTensor):
        memory, pos = self.swin(tensor_list.tensors)
        return memory, tensor_list.mask, pos


class GraphEmbeddingBackBone(nn.Module):
    def __init__(self, config):
        super(GraphEmbeddingBackBone, self).__init__()
        self.config = config
        if not config.node_resnet_froze:
            # for p in self.patchEmbed.parameters():
            #     p.requires_grad = False
            self.patchEmbed = PatchFeatureExtractor(config.patchEmbedding, device=config.device)
        if config.use_gin:
            self.gin = GIN(num_layers=config.num_layer, num_mlp_layers=config.num_mlp_per_layer,
                           input_dim=config.build_graph_dim, hidden_dim=config.graph_hidden_dim,
                           output_dim=config.graph_out_dim, final_dropout=0.5, learn_eps=config.learn_eps,
                           graph_pooling_type='mean', neighbor_pooling_type='mean', use_classification=config.use_classify,
                           tags=config.tag_num)
        self.imageEembed = tvmodels.resnet34(pretrained=True)
        if config.background_resnet_froze:
            for p in self.imageEembed.parameters():
                p.requires_grad = False
        self.imageEembed.fc = nn.Sequential() # remove classifier
        if config.use_pe:
            assert config.position_embedding in ("learned", "sine")
            if config.position_embedding == "learned":
                self.pos_embedding = PositionEmbeddingLearnedGraphNode(num_pos_feats=128)
            elif config.position_embedding == "sine":
                self.pos_embedding = PositionEmbeddingSineGraphNode(num_pos_feats=128, normalize=True)            
        self.num_channels = self.config.graph_hidden_dim
        self.device = config.device


    def forward(self, g, patches=None, image=None):
        classify_score = 0
        semantic = None
        if patches is None or image is None: #for caption visulization, given the patch features and image features calculated
            patch_embeddings = g.ndata['features']
            image_embedding = g.ndata['features'][0]
        else:
            if self.config.node_resnet_froze:
                patch_embeddings = g.ndata['feat'][:,:self.config.build_graph_dim]
            else:
                patch_embeddings = self.patchEmbed(patches)
            if self.config.use_gin:
                patch_embeddings, classify_score, semantic = self.gin(g, patch_embeddings) # 使用图神经网络
            # classify_score = 0
            image_embedding = self.imageEembed(image) # 加入图像背景信息
        # print(image_embedding.shape)
        g_unbatch = dgl.unbatch(g)
        batch_size = len(g_unbatch)
        if semantic is not None:
            semantic = semantic[:batch_size, :, :]
        cursor = 0
        features = torch.zeros(batch_size, self.config.max_node+1, self.num_channels).to(self.device)
        mask = torch.zeros((batch_size, self.config.max_node+1)).to(self.device)
        pos_x = torch.zeros((batch_size, self.config.max_node+1)).to(self.device)  ##relative position embedding
        pos_y = torch.zeros((batch_size, self.config.max_node+1)).to(self.device)
        ## TODO: 后续可以产生一个细胞重要度排名
        for i, g in enumerate(g_unbatch):
            # print(image_embedding)
            h_graph = patch_embeddings[cursor:cursor + g.num_nodes(), :]
            features[i, 0, :] = image_embedding[i]
            pos_x[i,0] = 0
            pos_y[i,0] = 0
            if g.num_nodes() <= self.config.max_node:
                features[i, 1:g.num_nodes()+1, :] = h_graph
                pos_x[i, 1:g.num_nodes()+1] = g.ndata['centroid'][:, 0]
                pos_y[i, 1:g.num_nodes()+1] = g.ndata['centroid'][:, 1]
                mask[i, g.num_nodes()+1:] = True
            else:
                features[i, 1:, :] = h_graph[:self.config.max_node, :]
                pos_x[i, 1:] = g.ndata['centroid'][:self.config.max_node, 0]
                pos_y[i, 1:] = g.ndata['centroid'][:self.config.max_node, 1]
            cursor += g.num_nodes()
        ## Sine position encoding TODO: relative pos embed
        if self.config.use_pe:
            pos_emd = self.pos_embedding(pos_x, pos_y)
        else:
            pos_emd = torch.zeros([batch_size, self.config.max_node, self.num_channels])
        return features, mask, pos_emd, classify_score, semantic

class GraphbackboneWoGlobal(nn.Module): ## 提取patch特征，不记录梯度，返回patch,无全局信息
    def __init__(self, config):
        super(GraphbackboneWoGlobal, self).__init__()
        self.config = config
        if not config.node_resnet_froze:
            self.patchEmbed = PatchFeatureExtractor(config.patchEmbedding, device=config.device)
            # for p in self.patchEmbed.parameters():
            #     p.requires_grad = False
        if config.use_gin:
            self.gin = GIN(num_layers=config.num_layer, num_mlp_layers=config.num_mlp_per_layer,
                           input_dim=config.build_graph_dim, hidden_dim=config.graph_hidden_dim,
                           output_dim=config.graph_out_dim, final_dropout=0.5, learn_eps=config.learn_eps,
                           graph_pooling_type='mean', neighbor_pooling_type='mean', use_classification=config.use_classify)
        if config.use_pe:
            self.pos_embedding = PositionEmbeddingSineGraphNode(num_pos_feats=128, normalize=True)
        self.num_channels = self.config.graph_hidden_dim
        self.device = config.device

    def forward(self, g, patches=None, image=None):
        classify_score = 0
        semantic = None
        if patches is None or image is None:  # for caption visulization, given the patch features and image features calculated
            patch_embeddings = g.ndata['features'][1:]
        else:
            if self.config.node_resnet_froze:
                # print(g.ndata)
                patch_embeddings = g.ndata['feat'][:, :self.config.build_graph_dim]
            else:
                patch_embeddings = self.patchEmbed(patches)
            if self.config.use_gin:
                patch_embeddings, classify_score, _ = self.gin(g, patch_embeddings)  # 使用图神经网络
        # print(image_embedding.shape)
        g_unbatch = dgl.unbatch(g)
        batch_size = len(g_unbatch)
        cursor = 0
        features = torch.zeros(batch_size, self.config.max_node, self.num_channels).to(self.device)
        mask = torch.zeros((batch_size, self.config.max_node)).to(self.device)
        pos_x = torch.zeros((batch_size, self.config.max_node)).to(self.device)  ##relative position embedding
        pos_y = torch.zeros((batch_size, self.config.max_node)).to(self.device)
        ## TODO: 后续可以产生一个细胞重要度排名
        for i, g in enumerate(g_unbatch):
            # print(image_embedding)
            h_graph = patch_embeddings[cursor:cursor + g.num_nodes(), :]
            if g.num_nodes() <= self.config.max_node:
                features[i, :g.num_nodes(), :] = h_graph
                if self.config.use_pe:
                    pos_x[i, :g.num_nodes()] = g.ndata['centroid'][:, 0]
                    pos_y[i, :g.num_nodes()] = g.ndata['centroid'][:, 1]
                mask[i, g.num_nodes() + 1:] = True
            else:
                features[i, :, :] = h_graph[:self.config.max_node, :]
                if self.config.use_pe:
                    pos_x[i, :] = g.ndata['centroid'][:self.config.max_node, 0]
                    pos_y[i, :] = g.ndata['centroid'][:self.config.max_node, 1]
            cursor += g.num_nodes()
        ## Sine position encoding TODO: relative pos embed
        if self.config.use_pe:
            pos_emd = self.pos_embedding(pos_x, pos_y)
        else:
            pos_emd = torch.zeros([batch_size, self.config.max_node, self.num_channels])
        return features, mask, pos_emd, classify_score, semantic

class GINBackBone(nn.Module):
    def __init__(self, config):
        super(GINBackBone, self).__init__()
        self.config = config
        self.gin = GIN(num_layers=config.num_layer, num_mlp_layers=config.num_mlp_per_layer,
                       input_dim=config.build_graph_dim, hidden_dim=config.graph_hidden_dim,
                       output_dim=config.graph_out_dim, final_dropout=0.5, learn_eps=config.learn_eps,
                       graph_pooling_type='mean', neighbor_pooling_type='mean', use_classification=config.use_classify)
        self.pos_embedding = PositionEmbeddingSineGraphNode(num_pos_feats=128, normalize=True)
        self.device = config.device
        self.num_channels = self.config.graph_hidden_dim

    def forward(self, g, h):
        h_update, classify_score, semantic = self.gin(g, h)
        h_update = h
        g_unbatch = dgl.unbatch(g)
        batch_size = len(g_unbatch)
        cursor = 0
        features = torch.zeros(batch_size, self.config.max_node, self.num_channels).to(self.device)
        mask = torch.zeros((batch_size, self.config.max_node)).to(self.device)
        pos_x = torch.zeros((batch_size, self.config.max_node)).to(self.device)  ##relative position embedding
        pos_y = torch.zeros((batch_size, self.config.max_node)).to(self.device)
        ## TODO: 后续可以产生一个细胞重要度排名
        for i, g in enumerate(g_unbatch):
            h_graph = h_update[cursor:cursor + g.num_nodes(), :]
            if g.num_nodes() <= self.config.max_node:
                features[i, :g.num_nodes(), :] = h_graph
                pos_x[i, :g.num_nodes()] = g.ndata['centroid'][:, 0]
                pos_y[i, :g.num_nodes()] = g.ndata['centroid'][:, 1]
                mask[i, g.num_nodes():] = True
            else:
                features[i, :, :] = h_graph[:self.config.max_node, :]
                pos_x[i, :] = g.ndata['centroid'][:self.config.max_node, 0]
                pos_y[i, :] = g.ndata['centroid'][:self.config.max_node, 1]
            cursor += g.num_nodes()
        ## Sine position encoding TODO: relative pos embed
        pos_emd = self.pos_embedding(pos_x, pos_y)
        return features, mask, pos_emd, classify_score, semantic

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        if self.num_channels != 2048:
            self.mlp = nn.Linear(self.num_channels, 2048)
            self.num_channels = 2048

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        # if self.mlp:
        #     xs = self.mlp(xs)
        for name, x in xs.items():
            # print(x.shape)
            if self.mlp:
                x = x.view(-1, 512, 49)
                x = x.permute(0, 2, 1)
                x = self.mlp(x)
                x = x.permute(0, 2, 1)
                x = x.view(-1, self.num_channels, 7, 7)
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(config):
    position_embedding = build_position_encoding(config)
    train_backbone = config.lr_backbone > 0
    return_interm_layers = False
    backbone = Backbone(config.backbone, train_backbone, return_interm_layers, config.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model