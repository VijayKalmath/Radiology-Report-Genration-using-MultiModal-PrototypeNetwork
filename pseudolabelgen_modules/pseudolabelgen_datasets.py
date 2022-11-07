import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle


class BaseDataset(Dataset):
    def __init__(self, args, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.split = split
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        with open(args.label_path, 'rb') as f:
            self.labels = pickle.load(f)

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        array = image_id.split('-')
        modified_id = array[0]+'-'+array[1]
        label = torch.FloatTensor(self.labels[modified_id])
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        sample = (image_id, image, label)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        label = torch.FloatTensor(self.labels[example['id']])
        if self.transform is not None:
            image = self.transform(image)
        sample = (image_id, image, label)
        return sample