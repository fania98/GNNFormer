from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization
from histocartography.preprocessing import NucleiExtractor
import torch
from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import sys
import os
import tqdm
from models import utils
from dataset import cellGraphCaptionWithPatch
from caption_configuration import GinConfig
from models.decoder import beam_search_decode, greedy_decode_graph
from transformers import BertTokenizer
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from PIL import Image
from models import caption as Caption
from nltk import word_tokenize

report_root = "Report/Images"
save_path = "Report/importance_cell_gradcam"
visualizer = OverlayGraphVisualization(
        instance_visualizer=InstanceImageVisualization(
            instance_style="fill"
        ),
        edge_style=None,
        colormap='jet',
        node_style='fill',
        show_colormap=True,
        min_max_color_normalize=True
    )

def visualize(graph, score, image_id, word, word_index):
    node_attributes = {}
    node_attributes["thickness"] = 1
    node_attributes["radius"] = 5
    node_attributes["color"] = score

    out = visualizer.process(
        canvas=Image.open(os.path.join(report_root, image_id+".png")),
        graph=graph,
        node_attributes=node_attributes,
        # instance_map=nuclei_map
    )
    out_fname = f"{image_id}_{word_index}.png"
    out.save(os.path.join(save_path, out_fname))


def main(config):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')
    nuclei_detector = NucleiExtractor()
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, criterion = Caption.build_model(config)
    caption_transformer = Caption.build_caption_transformer(model.transformer, model.input_proj1, model.mlp1)

    
    model.load_state_dict(torch.load("model_6_0.6000_3_3_36_resnet34_val0.pth")['model'])
    model.cuda()
    model.eval()
    caption_transformer.cuda()
    caption_transformer.eval()
    dataset_val = cellGraphCaptionWithPatch.build_dataset(config, mode='test', all_cap=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    image_num = 5
    def reshape_transform(tensor):
        ## tensor: [maxnode=601, batchsize=1, dim=256]
        result = tensor.permute(1,0,2)
        result = result.reshape(result.size(0),
                                result.size(1),result.size(2),1) ## [batchsize=1, maxnode=601, dim=256, 1]

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batchsize=1, channel=256, h=601, w=1]
        result = result.transpose(1, 2)
        return result

    grad_cam = GradCAMPlusPlus(model=caption_transformer,
                       target_layers=[caption_transformer.transformer.encoder.layers[-1]],
                       use_cuda=True,
                       reshape_transform=reshape_transform)

    print("Start generating..")

    for i in range(len(dataset_val)):
        scores = []
        # max = 0
        # min = 9999
        all_predicted_ids = []
        graph, patches, image, caps, cap_ids, image_id, cls = dataset_val.__getitem__(i)
        # img_original = Image.open(os.path.join(report_root, image_id+".png"))
        # nuclei_map, _ = nuclei_detector.process(img_original)
        graph = graph.to(device)
        patches = patches.to(device)
        image = image.to(device).unsqueeze(0)
        with torch.no_grad():
            features, mask, pos, classify, semantic = model.forward_backbone((graph, patches, image))
            print("output:", classify)
        caption, cap_mask = utils.create_caption_and_mask(101, config.max_position_embeddings, device)
        for i in range(config.max_position_embeddings - 1):
            with torch.no_grad():
                predictions = model.forward_transformer(features, mask, pos, caption, cap_mask, semantic)
                if config.trans_use_classify and (semantic is not None):
                    predictions = predictions[:, i+len(config.tag_num), :]
                else:
                    predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)
            if predicted_id[0] == 102:
                break
            cam = grad_cam(input_tensor=(features,mask,pos,caption, cap_mask),target_category=predicted_id[0].item())
            cam = cam.reshape(-1)
            scores.append(cam)
            all_predicted_ids.append(predicted_id[0])
            caption[:, i + 1] = predicted_id[0]
            cap_mask[:, i + 1] = False

        generate_caption = tokenizer.decode(all_predicted_ids)
        generate_caption_strs = generate_caption.split(".")
        print("max", max)
        print("min", min)
        # word_list = word_tokenize(generate_caption)
        print(generate_caption_strs)
        word_cache = []
        score = None
        score_num = 0
        word_index = 0
        for i in range(len(all_predicted_ids)):
            word_cache.append(all_predicted_ids[i])
            if score is None:
                score = scores[i]
            else:
                score += scores[i]
            score_num += 1
            cur_str = tokenizer.decode(word_cache)
            # word = tokenizer.decode(word_cache)
            # if word in word_list:
            if cur_str[-1] == ".":
                print(image_id, word_index)
                print(cur_str)
                new_score = []
                remove_nodes = []
                for i, s in enumerate(score[1:graph.num_nodes() + 1]):
                    if s > 0:
                        new_score.append(s)
                    else:
                        remove_nodes.append(i)
                # print(new_score)
                graph.remove_nodes(remove_nodes)
                if len(new_score) > 0:
                # visualize(graph, score[1:graph.num_nodes() + 1], image_id, cur_str, word_index)
                    visualize(graph, new_score, image_id, cur_str, word_index)
                score = None
                word_cache = []
                word_index += 1
                score_num = 0



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = GinConfig()

    main(config)

