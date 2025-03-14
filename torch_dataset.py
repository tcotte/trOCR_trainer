import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor

import evaluate

class HandWrittenDataset(Dataset):
    def __init__(self, root_dir :str, csv_path: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv_path)

        self.df = self.df[~self.df['number'].isnull()]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.df.iloc[index]['filename']))
        crop_image = image.crop(eval(self.df.iloc[index]['bbox']))
        return {'filename': self.df.iloc[index]['filename'],
                'image': crop_image,
                'target': int(self.df.iloc[index]['number'])}



class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=3):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # get file name + text
        image = Image.open(os.path.join(self.root_dir, self.df.iloc[index]['filename'])).convert('RGB')
        crop_image = image.crop(eval(self.df.iloc[index]['bbox']))
        pixel_values = self.processor(crop_image, return_tensors="pt").pixel_values

        text = self.df.iloc[index]['number']
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding




if __name__ == '__main__':
    metric = evaluate.load("cer")
    # dataset = HandWrittenDataset(root_dir='data', csv_path='crop_data.csv')
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    df = pd.read_csv('crop_data.csv')
    df = df[~df['number'].isnull()]

    train_df, test_df = train_test_split(df, test_size=0.2)
    print(len(train_df), len(test_df))


    train_dataset = IAMDataset(root_dir='data', df=train_df, processor=processor)
    encoding = train_dataset[0]

    for k, v in encoding.items():
        print(k, v.shape)

    labels = encoding['labels']
    print(labels)

    plt.imshow(torch.permute(encoding['pixel_values'], (1, 2, 0)))
    plt.show()

    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.decode(labels, skip_special_tokens=True)
    print('Decoded Label:', label_str)
