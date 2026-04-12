import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, images_dir, mask_dir):
        self.images = os.listdir(images_dir)
        self.images_dir = images_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        image = cv2.resize(image,(512,512))
        image = image.transpose(2,0,1)/255.0

        mask = cv2.imread(mask_path,0)
        mask = cv2.resize(mask,(512,512))

        return torch.tensor(image,dtype=torch.float32), torch.tensor(mask,dtype=torch.long)