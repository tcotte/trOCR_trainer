import logging
import os
import uuid
from typing import Dict

from picsellia import Client, Experiment
from picsellia.types.enums import LogType

from utils import get_GPU_occupancy


class PicselliaLogger:
    def __init__(self, client: Client, experiment: Experiment):
        self._client = client
        self._experiment = experiment


    def get_picsellia_experiment_link(self):
        client_id = self._client.id
        project_id = self.get_project_id_from_experiment()
        experiment_id = self._experiment.id

        link = f'https://app.picsellia.com/{str(client_id)}/project/{str(project_id)}/experiment/{experiment_id}'
        return link

    def get_project_id_from_experiment(self) -> uuid.UUID:
        for project in self._client.list_projects():
            for experiment in project.list_experiments():
                if str(experiment.id) == os.environ["experiment_id"]:
                    return project.id

    def log_split_table(self, annotations_in_split: Dict, title: str):
        data = {'x': [], 'y': []}
        for key, value in annotations_in_split.items():
            data['x'].append(key)
            data['y'].append(value)

        self._experiment.log(name=title, type=LogType.BAR, data=data)


    def on_train_begin(self):
        logging.info(f"Successfully logged to Picsellia\n You can follow experiment here: "
              f"{self.get_picsellia_experiment_link()} ")


    def on_epoch_end(self, epoch:int, train_loss:float, val_cer:float, display_gpu_occupancy:bool):
        self._experiment.log(name='Training loss', type=LogType.LINE, data=train_loss)

        logging.info(f"Epoch {epoch + 1}: Training loss {train_loss} / Validation CER: {val_cer}")
        self._experiment.log(name='Validation CER', type=LogType.LINE, data=val_cer)

        if display_gpu_occupancy:
            self._experiment.log(name='GPU occupancy (%)', type=LogType.LINE, data=get_GPU_occupancy())

    def store_model(self, model_path: str, model_name: str) -> None:
        self._experiment.store(model_name, model_path, do_zip=True)






