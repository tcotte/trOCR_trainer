import os

import torch
from torch.utils.data import Dataset
from PIL import Image

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(os.listdir(self.root_dir))

    # def get_item(self):

