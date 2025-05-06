import logging
import os
from typing import Union

import pandas as pd
from joblib import Parallel, delayed
from picsellia import DatasetVersion, Asset
from tqdm import tqdm


def get_dataframe_row(asset: Asset, classification_dataset_versions: Union[DatasetVersion, list[DatasetVersion]]) \
        -> None:
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
                                 num_workers: int=os.cpu_count()) -> pd.DataFrame:
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



