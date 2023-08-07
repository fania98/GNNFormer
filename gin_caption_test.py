import torch
from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import sys
import os
import tqdm
from models import utils, caption
from dataset import cellGraphCaptionWithPatch
from caption_configuration import GinConfig, SwinConfig
from models.decoder import beam_search_decode, greedy_decode_graph
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, accuracy_score
import evaluation
import math

@torch.no_grad()
def evaluate_metrics(model,config, criterion, dataloader, device, tokenizer, beam_size=1, output=False):
    model.eval()
    if output:
        with open("generation.csv","w") as f:
            pass
    criterion.eval()
    total = len(dataloader)
    num = 0
    gts, gen = {}, {}
    hypo_leision, gts_leision = [], []
    gt_labels = np.array([])
    pred_labels = np.array([])


    with tqdm.tqdm(total=total) as pbar:
        for ind, (graph, patches, images, caps, cap_ids, image_ids, clss) in enumerate(dataloader):
            graph = graph.to(device)
            patches = patches.to(device)
            images = images.to(device)

            if beam_size == 1:
                outputs, confidence, lengths, pred = greedy_decode_graph(model, graph, patches, images, device, config)
            else:
                outputs = beam_search_decode(model, graph_and_feats, device, config, beam_size)

            if config.use_classify:
                gt_labels = np.concatenate([gt_labels, clss.view(-1)])
                pred_labels = np.concatenate([pred_labels, pred])

            for s in range(len(caps)):
                hypo_str = tokenizer.decode(outputs[s][0], skip_special_tokens=True)
                print("hypo: "+hypo_str)
                gts[f"{ind}_{s}"] = caps[s]
                gen[f"{ind}_{s}"] = [hypo_str]
                num += 1
                if "high grade" in caps[s][0].lower():
                    gts_leision.append(0)
                else:
                    gts_leision.append(1)

                if "high grade" in hypo_str.lower():
                    hypo_leision.append(0)
                else:
                    hypo_leision.append(1)

                if output:
                    with open("generation.csv", "a") as f:
                        f.write(image_ids[s] + "\t")
                        f.write(hypo_str + "\t")
                        f.write(caps[s][0]+"\t")
                        f.write(str(clss[s].item())+"\t")
                        f.write(str(confidence[s][0][lengths[s]-2].item()))
                        f.write("\n")


            pbar.update(1)
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    leision_states = {}
    pcm = confusion_matrix(gts_leision, hypo_leision)
    tp, fn, fp, tn = pcm[0, 0], pcm[0, 1], pcm[1, 0], pcm[1, 1]
    leision_states['acc'] = (tp + tn) / (tn + fp + fn + tp)
    leision_states['sensitivity'] = tp / (fn + tp)
    leision_states['specificity'] = tn / (tn + fp)

    print(leision_states)

    scores, score = evaluation.compute_scores(gts, gen)
    print(scores)

    tag_states = {}
    if config.use_classify:
        pcm2 = confusion_matrix(gt_labels, pred_labels)
        print("tag pred: ")
        tp, fn, fp, tn = pcm2[0, 0], pcm2[0, 1], pcm2[1, 0], pcm2[1, 1]
        tag_states['acc'] = (tp + tn) / (tn + fp + fn + tp)
        tag_states['sensitivity'] = tp / (fn + tp)
        tag_states['specificity'] = tn / (tn + fp)
        print(tag_states)

    return scores, leision_states, tag_states




def main(config):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, criterion = caption.build_model(config)
    state_dict = torch.load("model_6_0.6105_3_3_36_resnet34_val2_wope.pth")["model"]
    model.load_state_dict(state_dict)
    model.cuda()
    dataset_val = cellGraphCaptionWithPatch.build_dataset(config, mode='test', all_cap=True)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers, collate_fn=cellGraphCaptionWithPatch.collate_fn_test)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("Start testing..")
    # for epoch in range(config.start_epoch, config.epochs):
    evaluate_metrics(model, config, criterion, data_loader_val, device, tokenizer, beam_size=1, output=True)



if __name__ == "__main__":
    import os
    config = GinConfig()