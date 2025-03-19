import logging
import os

import pandas as pd
from PIL import Image

from joblib import Parallel, delayed
from picsellia import Asset
from torch.utils.data import Dataset
from tqdm import tqdm


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
            logging.info(f'{filename} has no bounding box, it can not be annotated by the OCR model')

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