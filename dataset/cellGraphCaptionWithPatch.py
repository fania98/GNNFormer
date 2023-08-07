from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os
import torch
from dgl.data.utils import load_graphs
import dgl

from transformers import BertTokenizer

from dataset.utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype="float")
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    RandomRotation(),
    # tv.transforms.Lambda(under_max),
    tv.transforms.Resize([500, 500]),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    # tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = tv.transforms.Compose([
    # tv.transforms.Lambda(under_max),
    tv.transforms.Resize([500, 500]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class cellGraphCaptionWithPatch(Dataset):
    def __init__(self, root,  ann, max_length, limit, patch_size, graph_dir, transform=None, mode='training', all_cap=False, img_list=None):
        super().__init__()
        self.root = root
        self.root_graph = os.path.join(root, graph_dir)
        self.root_img = os.path.join(root, "Images")
        self.transform = transform
        self.mode = mode
        if img_list is None:
            self.annot = []
            for val in ann:
                for c in ann[val]['caption']:
                    self.annot.append((val, c, ann[val]["label"]))

        else:
            # img_ids = [val['id'] for val in ann['images'] if val['file_name'] in img_list]
            self.annot = []
            for val in ann:
                if self._process_img(val) in img_list:
                    for c in ann[val]['caption']:
                        self.annot.append((val, c, ann[val]["label"]))
        print(len(self.annot))
        self.id_captions, self.id_cls = self.get_img_caption_dict()
        self.patch_size = patch_size
        self.img_list = list(self.id_captions.keys())
        self.all_cap = all_cap
        # annot：1 个 image name对应一个caption的列表，每个image会重复5次，对应5个不同的caption
        # self.annot = self.annot[: limit]
        self.img_list = self.img_list[:limit]
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1

    def _process_graph(self, image_id):
        return image_id + ".bin"

    def _process_img(self, image_id):
        return image_id + ".png"

    def get_img_caption_dict(self):
        id_caption = {}
        id_cls = {}
        # keep a file for id_cls
        # if self.mode == "training":
        #     id_cls = read_json(os.path.join(self.root, "train_cls.json"))
        # else:
        #     id_cls = read_json(os.path.join(self.root, "test_cls.json"))

        for img_id, caption, cls in self.annot:
            # img_id = self._process(img_id)
            if img_id not in id_caption:
                id_caption[img_id] = []
                id_cls[img_id] = cls
            id_caption[img_id].append(caption)
        return id_caption, id_cls


    def __len__(self):
        if not self.all_cap:
            return len(self.annot)
        else:
            return len(self.img_list)

    def __getitem__(self, idx):
        if not self.all_cap:
            image_id, caption, _ = self.annot[idx]
            cls = self.id_cls[image_id]
            cell_graph = load_graphs(os.path.join(self.root_graph, self._process_graph(image_id)))[0][0] ## TODO : cellgraph's structure
            image = Image.open(os.path.join(self.root_img, self._process_img(image_id)))
            # TODO: graph data enhancement
            if self.transform:
                image_trans = self.transform(image)

            patches = torch.zeros(cell_graph.num_nodes(), 3, self.patch_size, self.patch_size)
            for i, c in enumerate(cell_graph.ndata['centroid']):
                min_x = max(0, int(c[0].item() - self.patch_size // 2))
                min_y = max(0, int(c[1].item() - self.patch_size // 2))
                max_x = min(int(c[0].item() + self.patch_size // 2), image_trans.shape[2])
                max_y = min(int(c[1].item() + self.patch_size // 2), image_trans.shape[1])
                # cell = image.crop((min_x, min_y, max_x, max_y))
                # cell.save(os.path.join(self.root, "cell_extract_examples",self._process_img(image_id+str(i))))
                patches[i, :, :max_y - min_y, :max_x - min_x] = image_trans[:, min_y:max_y, min_x:max_x]

            # image = nested_tensor_from_tensor_list(cell_graph)
            caption_encoded = self.tokenizer.encode_plus(
                caption, max_length=self.max_length, padding='max_length', return_attention_mask=True, return_token_type_ids=False, truncation=True)

            caption = np.array(caption_encoded['input_ids'])
            cap_mask = (
                1 - np.array(caption_encoded['attention_mask'])).astype(bool)
            # image_trans = tv.transforms.Resize([500, 500])(image_trans)
            return cell_graph, patches, image_trans, torch.tensor(caption), torch.tensor(cap_mask), cls
        else:
            image_id = self.img_list[idx]
            image = Image.open(os.path.join(self.root_img, self._process_img(image_id)))
            if self.transform:
                image_trans = self.transform(image) ## 大小没变
                # print(image_trans.shape)
            cls = self.id_cls[image_id]

            captions = self.id_captions[image_id]
            cell_graph = load_graphs(os.path.join(self.root_graph, self._process_graph(image_id)))[0][0]

            patches = torch.zeros(cell_graph.num_nodes(), 3, self.patch_size, self.patch_size)
            for i, c in enumerate(cell_graph.ndata['centroid']):
                min_x = max(0, int(c[0].item() - self.patch_size // 2))
                min_y = max(0, int(c[1].item() - self.patch_size // 2))
                max_x = min(int(c[0].item() + self.patch_size // 2), image_trans.shape[2])
                max_y = min(int(c[1].item() + self.patch_size // 2), image_trans.shape[1])
                # cell = image.crop((min_x, min_y, max_x, max_y))
                # cell.save(os.path.join(self.root, "cell_extract_examples",self._process_img(image_id+str(i))))
                patches[i, :, :max_y-min_y, :max_x-min_x] = image_trans[:, min_y:max_y, min_x:max_x]
            # image_trans = tv.transforms.Resize([500,500])(image_trans)
            captions_encoded_id = [self.tokenizer.encode_plus(
                c, max_length=self.max_length, padding='max_length', return_attention_mask=True,
                return_token_type_ids=False, truncation=True)['input_ids'] for c in captions]

            return cell_graph, patches, image_trans, captions, captions_encoded_id, image_id, cls


def collate_fn_train(batch):
    graphs = [item[0] for item in batch]
    graphs = dgl.batch(graphs)
    patches = torch.cat([item[1] for item in batch], dim=0)
    images = torch.stack([item[2] for item in batch])
    captions_encoded_ids = [item[3] for item in batch]
    captions_encoded_ids = torch.stack(captions_encoded_ids)
    cap_mask = [item[4] for item in batch]
    cap_mask = torch.stack(cap_mask)
    cls = [item[5] for item in batch]

    cls = torch.LongTensor(cls)
    return [graphs, patches, images, captions_encoded_ids, cap_mask, cls]

def collate_fn_test(batch):
    graphs = [item[0] for item in batch]
    graphs = dgl.batch(graphs)
    patches = torch.cat([item[1] for item in batch], dim=0)
    images = torch.stack([item[2] for item in batch])
    captions = [item[3] for item in batch]
    captions_encoded_ids = [item[4] for item in batch]
    img_ids = [item[5] for item in batch]
    cls = [item[6] for item in batch]
    # cls = []
    # for item in batch:
    #     cls += item[4]
    cls = torch.LongTensor(cls)
    return [graphs, patches, images, captions, captions_encoded_ids, img_ids, cls]


def build_dataset(config, mode='training', all_cap=False):
    root = config.dir
    if mode == 'training':
        train_file = os.path.join(
            root, 'train_annotation.json')
        data = cellGraphCaptionWithPatch(root, read_json(
            train_file), graph_dir=config.graph_dir, patch_size=config.patch_size, max_length=config.max_position_embeddings, limit=config.limit,
                           transform=train_transform, mode='training', all_cap=all_cap)

        return data

    elif mode == 'validation':
        val_file = os.path.join(
            config.dir, 'test_annotation.json')

        with open(os.path.join(root, config.val_file)) as f:
            img_list = [line.strip() for line in f.readlines()]

        data = cellGraphCaptionWithPatch(root, read_json(
            val_file), graph_dir=config.graph_dir, patch_size=config.patch_size, max_length=config.max_position_embeddings, limit=config.limit,
                           transform=val_transform, mode='validation', all_cap=all_cap, img_list=img_list)

        return data
    elif mode == 'test':
        val_file = os.path.join(
            config.dir, 'test_annotation.json')

        with open(os.path.join(root, config.test_file)) as f:
            img_list = [line.strip() for line in f.readlines()]

        data = cellGraphCaptionWithPatch(root, read_json(
            val_file), graph_dir=config.graph_dir, patch_size=config.patch_size, max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform,
                           mode='validation', all_cap=all_cap, img_list=img_list)

        return data

    else:
        raise NotImplementedError(f"{mode} not supported")

if __name__ == '__main__':

    from caption_configuration import GinConfig
    config = GinConfig()
    data = build_dataset(config,all_cap=True)
    data.__getitem__(2)