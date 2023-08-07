import torch
from torch import nn
import torch.nn.functional as F

from .utils import NestedTensor, nested_tensor_from_tensor_list
from .caption_backbone import build_backbone, GINBackBone, GraphEmbeddingBackBone, GraphbackboneWoGlobal, SwinTransformerBackBone
from .transformer import build_transformer
import dgl
from models.position_encoding import PositionEmbeddingSineGraphNode

class Caption(nn.Module):
    def __init__(self, backbone, transformer, hidden_dim, vocab_size, backbone_type):
        super().__init__()
        self.backbone = backbone
        print("backbone parameter num: ", sum(p.numel()
                                                 for p in self.backbone.parameters() if
                                                 p.requires_grad))
        self.backbone_type = backbone_type
        if self.backbone_type == 'gin' :
            self.input_proj1 = nn.Linear(
                backbone.num_channels, hidden_dim)
        elif self.backbone_type == 'cnn':
            self.input_proj1 = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        else:
            self.input_proj1 = nn.Linear(
                backbone.num_channels, hidden_dim)
        self.transformer = transformer
        print("transformer parameter num: ", sum(p.numel()
                       for p in self.transformer.parameters() if p.requires_grad))
        self.mlp1 = MLP(hidden_dim, 512, vocab_size, 1)
        print("last mlp parameter num: ", sum(p.numel()
                                                 for p in self.mlp1.parameters() if p.requires_grad))

    def forward_backbone(self, samples):
        if not isinstance(samples, NestedTensor) and self.backbone_type!='gin':
            samples = nested_tensor_from_tensor_list(samples)
        if self.backbone_type == 'gin':
            g, h, i = samples[0], samples[1], samples[2]
            features, mask, pos, classification, semantic = self.backbone(g, h, i) # b*512*(node个数+1)
            return features, mask, pos, classification, semantic
        elif self.backbone_type == 'cnn':
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()  ## list里只有一个，所以取 -1
            ## feature b*2048*19*19
            ## pos: b*256*19*19
            assert mask is not None
            return features, mask, pos
        else:
            memory, _, _ = self.backbone(samples)
            return memory, _, _

    def forward_transformer(self, features, mask, pos, target, target_mask, semantic=None):
        if self.backbone_type == 'gin':
            ## feature: b*200*514

            hs = self.transformer(self.input_proj1(features), mask,
                                  pos, target, target_mask, semantic)
            out = self.mlp1(hs.permute(1, 0, 2))
            return out
        elif self.backbone_type == 'cnn':
            src, mask = features[-1].decompose()  ## list里只有一个，所以取 -1
            hs = self.transformer(self.input_proj1(src), mask,
                                  pos[-1], target, target_mask)
            out = self.mlp1(hs.permute(1, 0, 2))
            return out
        else:
            # print(features.shape)
            mask = torch.zeros((features.shape[0], features.shape[1])).to(torch.bool).cuda()  # 8, 144

            hs = self.transformer(self.input_proj1(features).permute(1, 0, 2), mask,
                                  None, target, target_mask)

            out = self.mlp1(hs.permute(1, 0, 2))
            return out


    def forward(self, samples, target, target_mask):
        if not isinstance(samples, NestedTensor) and self.backbone_type!='gin':
            samples = nested_tensor_from_tensor_list(samples)
        if self.backbone_type == 'gin':
            g, h, i = samples[0], samples[1], samples[2]
            features, mask, pos, classify_score, semantic = self.backbone(g, h, i) # b*512*(node个数+1)
            ## feature: b*200*514
            hs = self.transformer(self.input_proj1(features), mask,
                                  pos, target, target_mask, semantic)
            out = self.mlp1(hs.permute(1, 0, 2))
            return out, classify_score

        elif self.backbone_type=='cnn':
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()  ## list里只有一个，所以取 -1
            ## feature b*2048*19*19
            ## pos: b*256*19*19
            assert mask is not None

            hs = self.transformer(self.input_proj1(src), mask,
                          pos[-1], target, target_mask)
            out = self.mlp1(hs.permute(1, 0, 2))
            return out, 0
        else:
            memory,_, _ = self.backbone(samples)
            mask = torch.zeros((memory.shape[0],memory.shape[1])).to(torch.bool).cuda() # 8, 144

            hs = self.transformer(self.input_proj1(memory).permute(1,0,2), mask,
                                  None, target, target_mask)

            out = self.mlp1(hs.permute(1, 0, 2))
            return out, 0



# class GinCaption(nn.Module):


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(config):
    if config.backbone == 'gin':
        if config.use_global_image:
            backbone = GraphEmbeddingBackBone(config)
        else:
            backbone = GraphbackboneWoGlobal(config)
        transformer = build_transformer(config, encoder=True)
        model = Caption(backbone, transformer, config.hidden_dim, config.vocab_size, backbone_type='gin')
    elif config.backbone == 'swin':
        backbone = SwinTransformerBackBone(config)
        transformer = build_transformer(config, encoder=False)
        model = Caption(backbone, transformer, config.hidden_dim, config.vocab_size, backbone_type='swin')
    else:
        backbone = build_backbone(config)
        transformer = build_transformer(config, encoder=True)
        model = Caption(backbone, transformer, config.hidden_dim, config.vocab_size, backbone_type='cnn')
    criterion = torch.nn.CrossEntropyLoss()
    return model, criterion


class caption_transformer(nn.Module):
    def __init__(self,transfomer,input_proj, mlp):
        super().__init__()
        self.input_proj1 = input_proj
        self.transformer = transfomer
        self.mlp1 = mlp

    def forward(self, input_tensors):
        features, mask, pos, target, target_mask = input_tensors[0],input_tensors[1],input_tensors[2],input_tensors[3],input_tensors[4]
        ## feature: b*node_num*graph_feature_length
        hs = self.transformer(self.input_proj1(features), mask,
                              pos, target, target_mask)
        out = self.mlp1(hs.permute(1, 0, 2))
        return out


def build_caption_transformer(transformer, input_proj, mlp):
    import copy
    transformer_n = copy.deepcopy(transformer)
    input_proj_n = copy.deepcopy(input_proj)
    mlp_n = copy.deepcopy(mlp)
    return caption_transformer(transformer_n, input_proj_n, mlp_n)

def graph_process(h_update, g, batch_size, num_channel,device, max_nodes=200):
    g_unbatch = dgl.unbatch(g)
    cursor = 0
    features = torch.zeros((batch_size, max_nodes, num_channel)).to(device)
    mask = torch.zeros((batch_size, max_nodes)).to(device)
    pos = torch.zeros((batch_size, max_nodes, 2)).to(device) ##TODO 变成sine,要和输入大小一样; relative position embedding
    # print(h_update.shape) ##  3649*514
    # h_update = h_update.permute((1,0)) ## 514*3649
    for i, g in enumerate(g_unbatch):
        h_graph = h_update[cursor:cursor+g.num_nodes(),:]
        if g.num_nodes() <= max_nodes:
            features[i, :g.num_nodes(),:] = h_graph
            # mask[i, :g.num_nodes(), :] = 1
            pos[i, :g.num_nodes(), :] = g.ndata['centroid']
        else:
            features[i, :, :] = h_graph[:max_nodes, :]
            # mask[i, :, :] = 1
            pos[i,:, :] = g.ndata['centroid'][:max_nodes, :]
        cursor += g.num_nodes()
    return features, mask, pos