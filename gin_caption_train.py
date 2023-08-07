import torch
from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import sys
import os
import tqdm
from models import utils, caption
from dataset import cellGraphCaption, cellGraphCaptionWithPatch
from caption_configuration import Config, GinConfig
import math
from utils import Logger
from gin_caption_test import evaluate_metrics
from transformers import BertTokenizer



def train_one_epoch(model, criterion, classify_criterion, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()
    classify_criterion.train()
    epoch_loss, epoch_generate_loss, epoch_classify_loss = 0.0, 0.0, 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for graph, patches, image, caps, cap_masks, cls in data_loader:
            graph = graph.to(device)
            patches = patches.to(device)
            image = image.to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            cls = cls.to(device)
            graph_patches_image = (graph, patches, image)
            outputs, classify = model(graph_patches_image, caps[:, :-1], cap_masks[:, :-1])
            if config.trans_use_classify:
                outputs = outputs[:, len(config.tag_num):, :]

            loss_generate = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_generate_value = loss_generate.item()
            epoch_generate_loss += loss_generate_value
            if config.use_classify:
                classify = classify[:patches.shape[0], :, :]
                classify = classify.permute(0, 2, 1)
                loss_classify = classify_criterion(classify, cls)

                loss = loss_classify * 0.1 + loss_generate
                loss_classify_value = loss_classify.item()
                epoch_classify_loss += loss_classify_value
                pbar.set_postfix(loss_generate=loss_generate_value, loss_classify=loss_classify_value, loss=loss.item())
            else:
                loss = loss_generate
                pbar.set_postfix(loss_generate=loss_generate_value, loss=loss.item())
            epoch_loss += loss.item()


            if not math.isfinite(loss.item()):
                print(f'Loss is {loss.item()}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)

    return epoch_loss / total, epoch_generate_loss/total, epoch_classify_loss/total


@torch.no_grad()
def evaluate(model, criterion, classify_criterion, data_loader, device):
    model.eval()
    criterion.eval()
    classify_criterion.train()
    validation_loss, v_generate_loss, v_classify_loss = 0.0, 0.0, 0.0
    total = len(data_loader)
    print(total)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with tqdm.tqdm(total=total) as pbar:
        for graph, patches, image, caps, cap_masks, cls in data_loader:
            graph = graph.to(device)
            caps = caps.to(device)
            image = image.to(device)
            cls = cls.to(device)
            cap_masks = cap_masks.to(device)
            graph_patches_images = (graph, patches, image)
            outputs, node_predictions = model(graph_patches_images, caps[:, :-1], cap_masks[:, :-1])
            output_words = torch.argmax(outputs, dim=-1)
            # print(output_words)
            words = tokenizer.decode(output_words[0], skip_special_tokens=True)
            # print(words)
            loss_generate = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_generate_value = loss_generate.item()
            v_generate_loss += loss_generate_value
            if config.use_classify:
                loss_classify = classify_criterion(node_predictions, cls)
                loss_classify_value = loss_classify.item()
                loss = loss_classify * 0.2 + loss_generate
                v_classify_loss += loss_classify_value
                pbar.set_postfix(loss_generate=loss_generate_value, loss_classify=loss_classify_value, loss=loss.item())
            else:
                loss = loss_generate
                pbar.set_postfix(loss_generate=loss_generate_value, loss=loss.item())
            validation_loss += loss.item()
            pbar.update(1)

    return validation_loss / total, v_generate_loss / total, v_classify_loss / total

def main(config):
    root = "img_seg_cap/"
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    logger = Logger(os.path.join(root,"train_log_stomach", "graph_caption"))
    prev_best_name = None
    model, criterion = caption.build_model(config)
    classify_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    model.to(device)
    state_dict = torch.load("weight493084032.pth")['model']
    model.load_state_dict(state_dict, strict=False)
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    param_dicts_name = [
        {"params": [n for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [n for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    print(param_dicts_name)

    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop, gamma=0.9)

    dataset_train = cellGraphCaptionWithPatch.build_dataset(config, mode='training')
    dataset_val = cellGraphCaptionWithPatch.build_dataset(config, mode='test')
    dataset_val_bleu = cellGraphCaptionWithPatch.build_dataset(config, mode='test', all_cap=True)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_val_bleu = torch.utils.data.SequentialSampler(dataset_val_bleu)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers, collate_fn=cellGraphCaptionWithPatch.collate_fn_train)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers, collate_fn=cellGraphCaptionWithPatch.collate_fn_train)
    data_loader_val_bleu = DataLoader(dataset_val_bleu, config.batch_size,
                                 sampler=sampler_val_bleu, drop_last=False, num_workers=config.num_workers, collate_fn=cellGraphCaptionWithPatch.collate_fn_test)
    if os.path.exists(config.checkpoint):
        print("Loading Checkpoint...")
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

    print("Start Training..")
    max_score = 0

    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")

        epoch_loss, e_generate_loss, e_classify_loss = train_one_epoch(
            model, criterion, classify_criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        if epoch <= 1:
            continue
        # validation_loss, v_generate_loss, v_classify_loss = evaluate(model, criterion, classify_criterion,  data_loader_val, device)
        scores, leision_state, tag_state = evaluate_metrics(model, config, criterion, data_loader_val_bleu, device, tokenizer,
                                                   beam_size=1)
        bleu4_score = scores["BLEU"][3]
        score = bleu4_score
        if score > max_score:
            max_score = score
            if prev_best_name != None:
                os.remove(os.path.join(root,"train_log_stomach","graph_caption", prev_best_name))
            model_name = os.path.join(root,"train_log_stomach","graph_caption",
                                      f'model_{epoch}_{bleu4_score:.4f}_{config.enc_layers}_{config.dec_layers}_{config.patch_size}_{config.patchEmbedding}_{config.val_file}.pth')
            prev_best_name = model_name
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, model_name)
            print(f"save {model_name}")

        logger.log(f"Epoch {epoch}: Training Loss: {epoch_loss}, Train_classify_loss: {e_classify_loss:.4f},train_generate_loss:{e_generate_loss:.4f}")
        # logger.log(f"\t Validation loss: {validation_loss}, val_classify_loss: {v_classify_loss:.4f},val_generate_loss:{v_generate_loss:.4f}")
        logger.log(f"\t BLEU {bleu4_score}; METEOR {scores['METEOR']}; ROUGE {scores['ROUGE']}; CIDEr {scores['CIDEr']};")


if __name__ == "__main__":
    # config = Config()
    config = GinConfig()
    main(config)
 