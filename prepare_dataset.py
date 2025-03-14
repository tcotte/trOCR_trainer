import os

import pandas as pd
from dotenv import load_dotenv
from picsellia import Client
from tqdm import tqdm

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


