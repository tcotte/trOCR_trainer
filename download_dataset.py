import os

from picsellia import Client
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv('.env')

    client = Client(os.getenv('PICSELLIA_TOKEN'), organization_name=os.getenv('ORGANIZATION_NAME'))
    dataset_version = client.get_dataset_version_by_id('019585ab-3cb0-7b59-9f6f-b74818844d2b')
    dataset_version.download('data')
