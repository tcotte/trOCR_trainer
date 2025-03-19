import logging
import os
import uuid
from collections import Counter
from typing import Dict

import picsellia
from picsellia import Client, Experiment, DatasetVersion
from picsellia.types.enums import LogType

from utils import get_GPU_occupancy


class PicselliaLogger:
    def __init__(self, client: Client, experiment: Experiment):
        """
        Logger class which enables to log metrics and store files on Picsellia.
        :param client: picsellia client
        :param experiment: experiment in which the metrics will be logged and files will be stored
        """
        self._client: Client = client
        self._experiment: Experiment = experiment


    def get_picsellia_experiment_link(self) -> str:
        """
        Get Picsellia experiment link
        :return: experiment link
        """
        client_id = self._client.id
        project_id = self.get_project_id_from_experiment()
        experiment_id = self._experiment.id

        link = f'https://app.picsellia.com/{str(client_id)}/project/{str(project_id)}/experiment/{experiment_id}'
        return link

    def get_project_id_from_experiment(self) -> uuid.UUID:
        """
        Retrieve project id from experiment id
        :return: project id
        """
        for project in self._client.list_projects():
            for experiment in project.list_experiments():
                if str(experiment.id) == os.environ["experiment_id"]:
                    return project.id

    def log_split_table(self, annotations_in_split: Dict, title: str) -> None:
        """
        Log table
        :param annotations_in_split:
        :param title:
        :return:
        """
        data = {'x': [], 'y': []}
        for key, value in annotations_in_split.items():
            data['x'].append(key)
            data['y'].append(value)

        self._experiment.log(name=title, type=LogType.BAR, data=data)


    def on_train_begin(self) -> None:
        """
        Do some actions when training begins:
        - Write experiment link in telemetry.
        - Plot labels of training/validation dataset version.
        :return:
        """
        logging.info(f"Successfully logged to Picsellia\n You can follow experiment here: "
              f"{self.get_picsellia_experiment_link()} ")

        self.plot_dataset_version_labels(dataset_version_names=['train', 'val'])


    def on_epoch_end(self, epoch:int, train_loss:float, val_cer:float, display_gpu_occupancy:bool) -> None:
        """
        Log training loss and validation CER when the epoch finishes
        :param epoch: epoch number which finished
        :param train_loss: previous training loss
        :param val_cer: previous validation CER
        :param display_gpu_occupancy: boolean which indicates if the GPU occupancy is displayed on Picsellia
        """
        self._experiment.log(name='Training loss', type=LogType.LINE, data=train_loss)

        logging.info(f"Epoch {epoch + 1}: Training loss {train_loss} / Validation CER: {val_cer}")
        self._experiment.log(name='Validation CER', type=LogType.LINE, data=val_cer)

        if display_gpu_occupancy:
            self._experiment.log(name='GPU occupancy (%)', type=LogType.LINE, data=get_GPU_occupancy())

    def store_model(self, model_path: str, model_name: str) -> None:
        """
        Store model as zip of files in Picsellia
        :param model_path: path of the folder that will be zipped and store in Picsellia
        :param model_name: name of the model on Picsellia
        """
        self._experiment.store(model_name, model_path, do_zip=True)

    def plot_dataset_version_labels(self, dataset_version_names: list[str]) -> None:
        """
        Plot label distribution of several dataset versions.
        :param dataset_version_names: Names of dataset versions
        """
        for version_name in dataset_version_names:
            self.dataset_label_distribution(dataset_version_name=version_name)

    def dataset_label_distribution(self, dataset_version_name: str) -> None:
        """
        Plot label distribution of dataset version in Picsellia.
        :param dataset_version_name: Alias of dataset version
        """
        list_label_names = []

        try:
            dataset_version: DatasetVersion = self._experiment.get_dataset(name=dataset_version_name)

            for annotation in dataset_version.list_annotations():
                list_label_names.append(annotation.list_classifications()[0].label.name)

            distribution_dict = Counter(list_label_names)
            data = {'x': list(distribution_dict.keys()), 'y': list(distribution_dict.values())}
            experiment.log(name=f'{dataset_version_name}_labels', type=LogType.BAR, data=data)

        except picsellia.exceptions.ResourceNotFoundError:
            logging.warning(f'Dataset version with name {dataset_version_name} was not found \n')

        except Exception as e:
            logging.warning(str(e))





if __name__ == '__main__':
    print(os.environ['api_token'])
    client = Client(api_token=os.environ['api_token'], organization_id=os.environ["organization_id"])
    experiment_id = '0195ada0-141a-793a-a176-b380b5bf2736'
    experiment = client.get_experiment_by_id(experiment_id)
    logger = PicselliaLogger(client=client, experiment=experiment)
    # ds_versions = experiment.list_attached_dataset_versions()
    # print(ds_versions)

    for ds_alias in ['train', 'val']:
        logger.dataset_label_distribution(dataset_version_name=ds_alias)

    # training_dataset_version = experiment.get_dataset(name='train')

    # training_dataset_version.list_labels()

