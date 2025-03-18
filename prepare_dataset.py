import logging
import os
from typing import Union

import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from picsellia import Client, DatasetVersion, Asset
from tqdm import tqdm

def get_dataframe_row(asset: Asset, classification_dataset_versions: Union[DatasetVersion, list[DatasetVersion]]):
    data = {}

    filename = asset.filename

    try:
        annotation = asset.list_annotations()[0]
        rectangle = annotation.list_rectangles()[0]
        xyxy_bbox = [rectangle.x, rectangle.y, rectangle.w + rectangle.x, rectangle.h + rectangle.y]

        data['filename'] = filename
        data['bbox'] = xyxy_bbox

        if isinstance(classification_dataset_versions, list):
            for dataset in classification_dataset_versions:
                found = False
                try:
                    classification_asset = dataset.find_asset(filename=filename)
                    number = classification_asset.list_annotations()[0].list_classifications()[0].label.name
                    data['number'] = number
                    found = True
                    break

                except Exception as e:
                    continue

            if not found:
                logging.warning(f'{filename} element was not found in any classification dataset')
                data['number'] = '?'

        else:
            try:
                classification_asset = classification_dataset_versions.find_asset(filename=filename)
                number = classification_asset.list_annotations()[0].list_classifications()[0].label.name
                data['number'] = number

            except Exception as e:
                logging.warning(f'{filename} element was not found in any classification dataset')
                data['number'] = None

    except IndexError:
        data['filename'] = filename
        data['bbox'] = None
        data['number'] = None

    return data


def create_dataframe_with_bboxes(dataset_object_detection_version: DatasetVersion,
                                 classification_dataset_versions: Union[DatasetVersion, list[DatasetVersion]],
                                 num_workers: int) -> pd.DataFrame:
    data = {
        'filename': [],
        'bbox': [],
        'number': []
    }

    logging.info('Prepare training dataframe...')

    list_rows_df = Parallel(n_jobs=num_workers)(delayed(get_dataframe_row)(asset, classification_dataset_versions) for
                                                asset in tqdm(dataset_object_detection_version.list_assets()))

    for row in list_rows_df:
        for key in row.keys():
            data[key].append(row[key])

    logging.info('Training dataframe was completed !')

    return pd.DataFrame(data)

if __name__ == '__main__':
    load_dotenv('.env')

    classification_dataset_ids = ['01957f94-5770-7608-b9bf-64fb4e1d45cf',
                                  '01955b42-5113-7ad0-9ddd-2723657cf660',
                                  '019541d8-d98a-7f15-9ea1-7b0e4944f635']

    client = Client(os.getenv('PICSELLIA_TOKEN'), organization_name=os.getenv('ORGANIZATION_NAME'))
    dataset_object_detection_version = client.get_dataset_version_by_id('019585ab-3cb0-7b59-9f6f-b74818844d2b')

    classification_datasets = [client.get_dataset_version_by_id(id=id_) for id_ in classification_dataset_ids]

    data = {
        'filename': [],
        'bbox': [],
        'number': []
    }

    for asset in tqdm(dataset_object_detection_version.list_assets()):
        filename = asset.filename

        try:
            annotation = asset.list_annotations()[0]
            rectangle = annotation.list_rectangles()[0]
            xyxy_bbox = [rectangle.x, rectangle.y, rectangle.w + rectangle.x, rectangle.h + rectangle.y]

            data['filename'].append(filename)
            data['bbox'].append(xyxy_bbox)

            for dataset in classification_datasets:
                found = False
                try:
                    classification_asset = dataset.find_asset(filename=filename)
                    number = classification_asset.list_annotations()[0].list_classifications()[0].label.name
                    data['number'].append(number)
                    found = True
                    break

                except Exception as e:
                    continue

            if not found:
                print(f'{filename} element was not found in any classification dataset')
                data['number'].append('?')

        except IndexError:
            data['filename'].append(filename)
            data['bbox'].append(None)
            data['number'].append(None)



    df = pd.DataFrame(data)
    df.to_csv('crop_data.csv', index=False)


