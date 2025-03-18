import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from joblib import Parallel, delayed
from picsellia import Asset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
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



class HandWrittenTrainDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=3):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        # get file name + text
        image = Image.open(os.path.join(self.root_dir, self.df.iloc[index]['filename'])).convert('RGB')

        bbox_coordinates = self.df.iloc[index]['bbox']
        if isinstance(bbox_coordinates, str):
            bbox_coordinates = eval(bbox_coordinates)

        crop_image = image.crop(bbox_coordinates)
        pixel_values = self.processor(crop_image, return_tensors="pt").pixel_values

        text = str(self.df.iloc[index]['number'])
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


class HandWrittenTestDataset(Dataset):
    def __init__(self, root_dir, object_detection_dataset_version, processor, num_workers:int = os.cpu_count(),
                 max_target_length=3):
        self.root_dir = root_dir
        self.__num_workers = num_workers
        self._object_detection_dataset_version = object_detection_dataset_version
        self.processor = processor
        self.max_target_length = max_target_length

        self._df = self.get_df_with_bounding_boxes()

    @staticmethod
    def complete_dataframe_row(asset: Asset) -> dict:
        data = {}

        filename = asset.filename

        try:
            annotation = asset.list_annotations()[0]
            rectangle = annotation.list_rectangles()[0]
            xyxy_bbox = [rectangle.x, rectangle.y, rectangle.w + rectangle.x, rectangle.h + rectangle.y]

            data['filename'] = filename
            data['bbox'] = xyxy_bbox

        except IndexError:
            data['filename'] = filename
            data['bbox'] = None

        return data

    def get_df_with_bounding_boxes(self) -> pd.DataFrame:
        logging.info('Prepare evaluation dataframe...')

        data = {
            'filename': [],
            'bbox': []
        }

        list_rows_df = Parallel(n_jobs=self.__num_workers)(
            delayed(self.complete_dataframe_row)(asset) for
            asset in tqdm(self._object_detection_dataset_version.list_assets()))

        for row in list_rows_df:
            for key in row.keys():
                data[key].append(row[key])


        df = pd.DataFrame(data)
        logging.info('Evaluation dataframe was completed !')

        return df[~df['bbox'].isnull()]

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        # get file name + text
        image = Image.open(os.path.join(self.root_dir, self._df.iloc[index]['filename'])).convert('RGB')

        bbox_coordinates = self._df.iloc[index]['bbox']
        if isinstance(bbox_coordinates, str):
            bbox_coordinates = eval(bbox_coordinates)

        crop_image = image.crop(bbox_coordinates)
        pixel_values = self.processor(crop_image, return_tensors="pt").pixel_values

        encoding = {"pixel_values": pixel_values.squeeze(), 'filename': self._df.iloc[index]['filename']}
        return encoding




if __name__ == '__main__':
    metric = evaluate.load("cer")
    # dataset = HandWrittenDataset(root_dir='data', csv_path='crop_data.csv')
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    df = pd.read_csv('../../crop_data.csv')
    df = df[~df['number'].isnull()]

    train_df, test_df = train_test_split(df, test_size=0.2)
    print(len(train_df), len(test_df))


    train_dataset = HandWrittenTrainDataset(root_dir='data', df=train_df, processor=processor)
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
