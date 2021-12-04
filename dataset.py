import cv2
import random
from glob import glob
import torch 
import numpy as np
from torch.utils.data import Dataset

class ArtData(Dataset):
    def __init__(self, images_dir, resolution=256, limit=2048, transform=None):
        super(ArtData, self).__init__()
        image_paths = glob(f"{images_dir}/*.jpg")
        image_paths.sort()
        random.shuffle(image_paths)
        num_images = min(limit, len(image_paths))
        self.image_paths = sorted(image_paths[:num_images])
        self.transform = transform
        self.res = resolution

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.res, self.res))
        if self.transform:
            return self.transform(image)
        return image
