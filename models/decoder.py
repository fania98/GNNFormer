from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
import torch
from torch import nn
# from fairseq.models.transformer import transformer_decoder,transformer_config
from .utils import create_caption_and_mask
from .utils import BeamSearchNode
from queue import PriorityQueue
from copy import deepcopy
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, ):
        super(Decoder, self).__init__()
        config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
        config.is_decoder = True
        self.backbone = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder', config=config)
        # config = transformer_config.DecoderConfig
        # config.learned_pos = True
        # config.normalize_before = True
        # self.backbone = transformer_decoder(config,dictionary,embed_tokens)

    # def forward(self, prev_output_tokens, encoder_out):
    #     self.backbone()

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask, labels):
        outputs = self.backbone(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask)
        print(outputs)
        return outputs.logits


@torch.no_grad()
def greedy_decode_graph(model, graph, patches, images, device, config):
    import dgl
    results = torch.zeros(config.batch_size, 1, config.max_position_embeddings)
    confidences = torch.zeros(config.batch_size, 1, config.max_position_embeddings)
    lengths = []
    g_unbatch = dgl.unbatch(graph)
    nums = [g.num_nodes() for g in g_unbatch]
    cursor = 0
    pred_labels = np.array([])
    for ind, g in enumerate(g_unbatch):
        g_patch = patches[cursor:cursor+g.num_nodes(),:, :, :]
        caption, cap_mask = create_caption_and_mask(101, config.max_position_embeddings, device)
        features, mask, pos, classify_score, semantic = model.forward_backbone((g,g_patch,images[ind].unsqueeze(0)))
        if config.use_classify:
            classify_score = classify_score[0, :, :]
            classify_pred = torch.argmax(classify_score, dim=-1)
            # print("pred: ", classify_pred)
            pred_labels = np.concatenate([pred_labels, classify_pred.cpu().numpy().flatten()])
        length = 0
        for i in range(config.max_position_embeddings - 1):
            predictions = model.forward_transformer(features, mask, pos, caption, cap_mask, semantic)
            if config.trans_use_classify and (semantic is not None):
                predictions = predictions[:, i + len(config.tag_num), :]
            else:
                predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)
            if predicted_id[0] == 102:
                length = i
                break

            caption[:, i + 1] = predicted_id[0]
            cap_mask[:, i + 1] = False
            predictions = torch.softmax(predictions, dim=-1)
            # confidence for giving "high grade" as result
            # confidences[ind, 0, i+1] = predictions[0][predicted_id[0]]
            confidences[ind, 0, i + 1] = predictions[0][2152]
        if length == 0:
            length = config.max_position_embeddings - 1
        lengths.append(length)
        results[ind, :,: ] = caption
        cursor += g.num_nodes()
    # print(results.shape)
    # print(results)
    return results, confidences, lengths, pred_labels


@torch.no_grad()
def greedy_decode(model, images, device, config):
    results = torch.zeros(config.batch_size, 1, config.max_position_embeddings)
    for ind, image in enumerate(images):
        image = torch.unsqueeze(image, 0).to(device)
        features, mask, pos = model.forward_backbone(image)
        caption, cap_mask = create_caption_and_mask(101, config.max_position_embeddings, device)
        for i in range(config.max_position_embeddings - 1):
            predictions = model.forward_transformer(features,mask, pos, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)
            if predicted_id[0] == 102:
                break
            caption[:, i + 1] = predicted_id[0]
            cap_mask[:, i + 1] = False
        results[ind, :,: ] = caption
    # print(results.shape)
    # print(results)
    return results

def beam_search_decode(model, images, device, config, beam_size=4, required_num=1):
    # results = torch.zeros(config.batch_size, config.vocab_size, config.max_position_embeddings)
    end_captions_all = torch.empty(0).to(device)
    for ind, image in enumerate(images):
        end_caption = torch.empty(0).to(device)
        # print(ind)
        image = torch.unsqueeze(image,0).to(device)
        caption, cap_mask = create_caption_and_mask(101, config.max_position_embeddings, device)
        node = BeamSearchNode(caption, cap_mask,  0, 1, 101)
        nodes = PriorityQueue()
        # start the queue
        nodes.put((-node.logprob, node))
        for i in range(config.max_position_embeddings - 1):
            find_end = False
            next_nodes = PriorityQueue()
            # print("position ",i)
            for b in range(beam_size):
                if i == 0 and b > 0:
                    continue
                caption_node = nodes.get()[1]
                # print(caption_node.caption)
                if caption_node.wordid == 102:
                    end_caption = torch.cat([end_caption, caption_node.caption], dim=0)
                    # break
                    if end_caption.shape[0] >= required_num:
                        # end_caption = torch.tensor(end_caption)
                        end_caption = end_caption.unsqueeze(0)
                        end_captions_all = torch.cat([end_captions_all, end_caption], dim=0)
                        # end_captions_all.append(end_caption)
                        find_end = True
                        break
                predictions = model(image, caption_node.caption, caption_node.caption_mask)
                predictions = predictions[:, i, :]
                predictions = torch.softmax(predictions, dim=-1)
                predictions = torch.log(predictions) # <0
                predicted_prob, predicted_id = torch.topk(predictions,largest=True, k=beam_size, dim=-1)
                # print(f"i: {i}")
                # print("predicted_id: ",predicted_id)
                for nb in range(beam_size):
                    # print(predicted_prob[0][nb], predicted_id[0][nb])
                    new_node = BeamSearchNode(deepcopy(caption_node.caption).to(device), deepcopy(caption_node.caption_mask).to(device), 0, i+2, predicted_id[0][nb])
                    new_node.caption[:, i + 1] = predicted_id[0][nb]
                    # print(new_node.caption)
                    new_node.caption_mask[:, i + 1] = False
                    new_node.logprob = predicted_prob[0][nb] + caption_node.logprob
                    next_nodes.put((-new_node.logprob,new_node))
            if find_end:
                break

            # clean the low prob sentences
            nodes = PriorityQueue()
            for ind in range(beam_size):
                nodes.put(next_nodes.get())

            # print(nodes.get()[1].caption)

        # results[ind, :,: ] = nodes.get()[1].caption
        # print(nodes.get()[1].caption)
    # print(end_captions_all)
    return end_captions_all