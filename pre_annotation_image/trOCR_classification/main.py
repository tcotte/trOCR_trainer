import logging
import os
import sys
from datetime import datetime

logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.info(sys.version)

import cv2
import torch
from picsellia import Client, DatasetVersion
from picsellia.types.enums import InferenceType
from torch import FloatTensor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel




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


def annotate_asset(classification_dataset_version: DatasetVersion, filename: str, model_output: str) -> None:
    """
    Create classification annotation to an asset. Label should be a digit.
    :param classification_dataset_version: dataset version where the asset will be annotated
    :param filename: filename of the asset which will be annotated
    :param model_output: model predictions output: must be converted to a digit to be valid
    """
    try:
        int(model_output)
        # find asset
        input_asset = classification_dataset_version.find_asset(filename=filename)

        # delete annotation if it already exists one
        if asset.list_annotations() != []:
            asset.delete_annotations()

        # create annotation
        annotation = input_asset.create_annotation()
        label = classification_dataset_version.get_or_create_label(name=model_output)
        annotation.create_classification(label=label)
        logging.info(f'{filename} annotated with {label} digit')

    except ValueError:
        logging.warning(f'{filename} - {model_output} can not be converted to integer format')


def output_model_postprocessing(scores: tuple[FloatTensor]) -> list[str]:
    """
    Post-process the prediction of the *VisionEncoderDecoderModel* to get a list of identified number.
    :param scores: scores of each detected characters.
    :return: list of detected numbers
    """
    str_number = []
    last_str_number = ''
    for score in generated_ids.scores:
        if score.is_cuda:
            score = score.detach().cpu()

        if torch.argmax(score) in list(digit_ids.values()):
            # if there is digit redundancy -> cancel the process and return the processed output
            if last_str_number == invert_digit_ids[int(torch.argmax(score))]:
                return str_number

            else:
                str_number.append(invert_digit_ids[int(torch.argmax(score))])
                last_str_number = invert_digit_ids[int(torch.argmax(score))]

        elif (int(torch.argmax(score)) == 2) or (int(torch.argmax(score)) == 479):  # space or dot character
            pass

        else:
            # get maximum confidence of digit's scores
            indices = torch.tensor(list(digit_ids.values()))
            str_number.append(str(int(torch.argmax(torch.index_select(score, 1, indices)))))
            return str_number

    return str_number

def configure_processor(model_parameters: dict, processing_context: dict) -> TrOCRProcessor:
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
    output_dataset_version_id = context["output_dataset_version_id"]
    output_dataset_version = client.get_dataset_version_by_id(output_dataset_version_id)

    # if fine-tuned model
    model_version_id = context["model_version_id"]
    model_version = client.get_model_version_by_id(model_version_id)
    model_parameters: dict = model_version.get_context().parameters



    logging.info(f'Annotations will be created in dataset version {output_dataset_version.name}')

    datalake = client.get_datalake()
    object_detection_dataset_version = client.get_dataset_version_by_id(input_dataset_version_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # classification_dataset_version = create_dataset_version_from_another(
    #     input_dataset_version=object_detection_dataset_version)
    classification_dataset_version = output_dataset_version
    # copy images of object detection dataset version to classification dataset version
    assets = object_detection_dataset_version.list_assets()
    data = assets.as_list_of_data()
    output_dataset_version.add_data(data)

    # get pretrained encoder/decoder model
    # define encoder/decoder model
    model = configure_encoder_decoder_model(model_parameters=model_parameters, processing_context=context)

    # define processor
    processor = configure_processor(model_parameters=model_parameters, processing_context=context)

    # try to retrieve model weights
    try:
        model_version.get_file('model-best')
    except:
        logging.warning('Can not retrieve model-best file')

    model.to(device)

    tokenizer = processor.tokenizer

    vocab = processor.tokenizer.get_vocab()

    # Extract token IDs for digits (0-9)
    digit_ids = {str(i): vocab[str(i)] for i in range(300)}

    # invert dictionary
    invert_digit_ids = {v: k for k, v in digit_ids.items()}
    invert_vocab = {v: k for k, v in vocab.items()}

    data_path: str = 'data'
    object_detection_dataset_version.download(data_path)

    for filename in os.listdir(data_path):
        asset = object_detection_dataset_version.find_asset(filename=filename)
        annotation = asset.get_annotation()

        # asset is supposed to contain only one rectangle
        if not annotation.list_rectangles() == []:
            rectangle = annotation.list_rectangles()[0]

            img = cv2.imread(os.path.join(data_path, filename))
            img_crop = img[rectangle.y:rectangle.y + rectangle.h, rectangle.x:rectangle.x + rectangle.w]

            pixel_values = processor(images=img_crop, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            generated_ids = model.generate(pixel_values,
                                           max_length=5,  # Max length of the generated sequence
                                           do_sample=True,  # We disable sampling to use greedy decoding
                                           output_scores=True,
                                           return_dict_in_generate=True)

            str_number = output_model_postprocessing(generated_ids.scores)

            if str_number != []:
                logging.info(str_number)
                str_number = ''.join(str_number)

                annotate_asset(classification_dataset_version=classification_dataset_version,
                               filename=filename,
                               model_output=str_number)
            else:
                logging.warning(f'Not any numbers were detected in the picture {filename}')

        else:
            logging.warning(f'Can not label {filename} because there is not any annotated rectangle')
