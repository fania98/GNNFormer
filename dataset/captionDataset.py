from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os
import torch

from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 224


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
    tv.transforms.Resize([MAX_DIM, MAX_DIM]),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    # tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

val_transform = tv.transforms.Compose([
    # tv.transforms.Lambda(under_max),
    tv.transforms.Resize([MAX_DIM, MAX_DIM]),
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])


class captionDataset(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training', all_cap=False, img_list=None):
        super().__init__()

        self.root = root
        self.transform = transform
        self.mode = mode
        if img_list is None:
            self.annot = []
            for val in ann:
                for c in ann[val]['caption']:
                    self.annot.append((self._process(val), c, ann[val]["label"]))

        else:
            # img_ids = [val['id'] for val in ann['images'] if val['file_name'] in img_list]
            self.annot = []
            for val in ann:
                if self._process(val) in img_list:
                    for c in ann[val]['caption']:
                        self.annot.append((self._process(val), c, ann[val]["label"]))
        print(len(self.annot))
        self.id_captions, self.id_cls = self.get_img_caption_dict()
        self.img_list = list(self.id_captions.keys())
        self.all_cap = all_cap
    
        self.img_list = self.img_list[: limit]

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1

    def _process(self, image_id):
        return image_id + ".png"

    def get_img_caption_dict(self):
        id_caption = {}
        id_cls = {}
        for img_id, caption, clss in self.annot:
            # img_id = self._process(img_id)
            if img_id not in id_caption:
                id_caption[img_id] = []
            id_caption[img_id].append(caption)
            id_cls[img_id] = clss
        return id_caption, id_cls


    def __len__(self):
        if not self.all_cap:
            return len(self.annot)
        else:
            return len(self.img_list)

    def __getitem__(self, idx):
        if not self.all_cap:
            image_id, caption,_ = self.annot[idx]
            image = Image.open(os.path.join(self.root, image_id))

            if self.transform:
                image = self.transform(image)
            image = nested_tensor_from_tensor_list(image.unsqueeze(0))
            caption_encoded = self.tokenizer.encode_plus(
                caption, max_length=self.max_length, padding='max_length', return_attention_mask=True, return_token_type_ids=False, truncation=True)

            caption = np.array(caption_encoded['input_ids'])
            cap_mask = (
                1 - np.array(caption_encoded['attention_mask'])).astype(bool)
            ## image mask在image部分的值是false,padding的部分值是true
            return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask
        else:
            image_id = self.img_list[idx]
            captions = self.id_captions[image_id]
            image = Image.open(os.path.join(self.root, image_id))
            if self.transform:
                image = self.transform(image)
            image = nested_tensor_from_tensor_list(image.unsqueeze(0))
            # print(captions)
            captions_encoded_id = [self.tokenizer.encode_plus(
                c, max_length=self.max_length, padding='max_length', return_attention_mask=True,
                return_token_type_ids=False, truncation=True)['input_ids'] for c in captions]
            # captions = np.array(caption_encoded['input_ids'])
            # print(captions_encoded_id)
            # cap_mask = (
            #         1 - np.array(caption_encoded['attention_mask'])).astype(bool)

            return image.tensors.squeeze(0), image.mask.squeeze(0), captions, captions_encoded_id, image_id, self.id_cls[image_id]


def collate_fn(batch):
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    captions_encoded_ids = [item[3] for item in batch]
    images, masks = torch.stack(images), torch.stack(masks)
    captions = [item [2] for item in batch]
    img_ids = [item[4] for item in batch]
    img_cls = [item[5] for item in batch]
    return [images, masks, captions, captions_encoded_ids, img_ids, img_cls]




def build_dataset(config, mode='training', all_cap=False):
    root = config.dir
    if mode == 'training':
        train_file = os.path.join(
            root, 'train_annotation.json')
        data = captionDataset(os.path.join(root,"Images"), read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit,
                           transform=train_transform, mode='training', all_cap=all_cap)

        return data

    elif mode == 'validation':
        val_file = os.path.join(
            config.dir, 'test_annotation.json')

        with open(os.path.join(root, config.val_file)) as f:
            img_list = [line.strip() for line in f.readlines()]

        data = captionDataset(os.path.join(root,"Images"), read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit,
                           transform=val_transform, mode='validation', all_cap=all_cap, img_list=img_list)

        return data
    elif mode == 'test':
        val_file = os.path.join(
            config.dir, 'test_annotation.json')

        with open(os.path.join(root, config.test_file)) as f:
            img_list = [line.strip() for line in f.readlines()]

        data = captionDataset(os.path.join(root,"Images"), read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform,
                           mode='validation', all_cap=all_cap, img_list=img_list)



        return data

    else:
        raise NotImplementedError(f"{mode} not supported")

if __name__ == '__main__':

    from caption_configuration import Config
    config = Config()
    data = build_dataset(config,all_cap=True)
    data.__getitem__(2)