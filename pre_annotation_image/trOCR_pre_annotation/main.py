import logging
import os
import sys
import zipfile
from datetime import datetime
from typing import Union

from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.info(sys.version)

import torch
from picsellia import Client, DatasetVersion, Dataset
from picsellia.types.enums import InferenceType
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from utils import HandWrittenTestDataset



def create_dataset_version_from_another(input_dataset_version: DatasetVersion,
                                        inference_type: InferenceType = InferenceType.CLASSIFICATION) -> DatasetVersion:
    """
    Create a new dataset version importing data from another dataset version.
    :param input_dataset_version: dataset version which will copied (data from this dataset version will be copied to a
    new dataset version)
    :param inference_type: inference type of the new dataset version
    :return: new dataset version
    """
    # create new dataset version
    dataset = client.get_dataset_by_id(input_dataset_version.origin_id)

    date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dataset_version = dataset.create_version(f'{inference_type.value}-{date_time}')
    output_dataset_version.set_type(type=inference_type)

    # copy data from input dataset version
    assets = input_dataset_version.list_assets()
    data = assets.as_list_of_data()
    output_dataset_version.add_data(data)

    return output_dataset_version


def annotate_asset(classification_dataset_version: DatasetVersion, filenames: Union[str, list[str]],
                   model_outputs: Union[str, list[str]]) -> None:
    """
    Create classification annotation to an asset. Label should be a digit.
    :param classification_dataset_version: dataset version where the asset will be annotated
    :param filenames: filename(s) of asset(s) which will be annotated
    :param model_outputs: model prediction output(s)
    """
    if isinstance(model_outputs, str):
        model_outputs = [model_outputs]

    if isinstance(filenames, str):
        filenames = [filenames]

    for model_output, filename in zip(model_outputs, filenames):
        # find asset
        input_asset = classification_dataset_version.find_asset(filename=filename)

        # delete annotation if it already exists one
        if input_asset.list_annotations() != []:
            input_asset.delete_annotations()

        # create annotation
        annotation = input_asset.create_annotation()
        label = classification_dataset_version.get_or_create_label(name=model_output)
        annotation.create_classification(label=label)
        logging.info(f'{filename} annotated with {label.name}')



def configure_processor(model_parameters: dict, processing_context: dict) -> TrOCRProcessor:
    """
    Configure processor depending on the parameters of:
    - The model version
    - The parameters sent in the processing (if the model version was not selected)
    If not any argument concerning the processor was passed, the processor will be *microsoft/trocr-base-handwritten* as
    default.
    :param model_parameters: Parameters of the deployed model version used to do the processing.
    :param processing_context: Parameters sent to the processing.
    :return: Processor which will be used to
    """
    if 'pretrained_trOCRprocessor' in model_parameters.keys():
        return TrOCRProcessor.from_pretrained(model_parameters['pretrained_trOCRprocessor'])

    elif 'pretrained_trOCRprocessor' in processing_context.keys():
        try:
            return TrOCRProcessor.from_pretrained(processing_context['pretrained_trOCRprocessor'])
        except OSError as e:
            raise e

    else:
        return TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')

def configure_encoder_decoder_model(model_parameters: dict, processing_context: dict) -> VisionEncoderDecoderModel:
    if 'vision_encoder_decoder_model_weights' in model_parameters.keys():
        return VisionEncoderDecoderModel.from_pretrained(model_parameters['vision_encoder_decoder_model_weights'])

    elif 'vision_encoder_decoder_model_weights' in processing_context.keys():
        try:
            return VisionEncoderDecoderModel.from_pretrained(processing_context['vision_encoder_decoder_model_weights'])
        except OSError as e:
            raise e

    else:
        return VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-stage1')


if __name__ == '__main__':
    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]
    job_id = os.environ["job_id"]

    client = Client(api_token=api_token, organization_id=organization_id)

    job = client.get_job_by_id(job_id)

    context = job.sync()["dataset_version_processing_job"]
    input_dataset_version_id = context["input_dataset_version_id"]


    # output_dataset_version_id = context["output_dataset_version_id"]
    # output_dataset_version = client.get_dataset_version_by_id(output_dataset_version_id)

    if 'max_length_characters' in context.keys():
        max_length_characters = context["max_length_characters"]
    else:
        max_length_characters = 5
        logging.info(f'max_length_characters not found: {max_length_characters} is set by default.')

    if 'batch_size' in context.keys():
        batch_size = context['batch_size']
    else:
        batch_size = 4
        logging.info(f'batch size not found: {batch_size} is set by default.')

    # if fine-tuned model
    model_version_id = context["model_version_id"]
    model_version = client.get_model_version_by_id(model_version_id)
    model_parameters: dict = model_version.get_context().parameters

    logging.info('Parameters used for the training: ')
    for k, v in context.items():
        logging.info(f'     - {k}: {v}')


    object_detection_dataset_version = client.get_dataset_version_by_id(input_dataset_version_id)

    # create dataset
    classification_dataset_version, fork_job = object_detection_dataset_version.fork(
        version=f'classification_{object_detection_dataset_version.name}',
        description=f'Forked from version {object_detection_dataset_version.name} via trOCR_preannotation processing',
        type=InferenceType.CLASSIFICATION,
        wait=True
    )

    # dataset: Dataset = client.get_dataset_by_id(object_detection_dataset_version.origin_id)
    # classification_dataset_version = dataset.create_version(f'classification_{object_detection_dataset_version.name}')
    # classification_dataset_version.set_type(type=InferenceType.CLASSIFICATION)
    # assets = object_detection_dataset_version.list_assets()
    # data = assets.as_list_of_data()
    # classification_dataset_version.add_data(data)


    logging.info(f'{classification_dataset_version.name} was created successfully thanks to the fork of '
                 f'{object_detection_dataset_version.name}')


    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # define processor
    processor = configure_processor(model_parameters=model_parameters, processing_context=context)

    # try to retrieve model weights
    try:
        model_version.get_file('model-best')
        # download model
        logging.info('Downloading the model...')
        model_path: str = 'models'
        model_version.get_file('model-best').download(model_path)
        logging.info('Model was successfully downloaded !')

        logging.info('Extracting model zip...')
        with zipfile.ZipFile(os.path.join(model_path, 'model.zip'), 'r') as zip_ref:
            zip_ref.extractall(model_path)
        logging.info('Extraction was done !')

        model = VisionEncoderDecoderModel.from_pretrained('models')


    except:
        model = configure_encoder_decoder_model(model_parameters=model_parameters, processing_context=context)
        logging.warning('Can not retrieve model-best file')



    model.to(device)
    model.eval()

    data_path: str = 'data'
    object_detection_dataset_version.download(data_path)

    test_dataset = HandWrittenTestDataset(root_dir=data_path,
                                          object_detection_dataset_version=object_detection_dataset_version,
                                          processor=processor,
                                          max_target_length=max_length_characters)

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(test_data_loader):
        with torch.no_grad():
            outputs = model.generate(batch["pixel_values"].to(device))
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
        annotate_asset(classification_dataset_version=classification_dataset_version,
                       filenames=batch['filename'],
                       model_outputs=generated_text)

    logging.info('Pre annotation with trOCR was successfully completed')
