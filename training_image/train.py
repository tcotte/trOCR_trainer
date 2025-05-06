import logging
import os
import shutil
from typing import Optional
from uuid import UUID

import evaluate
import picsellia.exceptions
import torch
from joblib import Parallel, delayed
from picsellia import Client, DatasetVersion, Asset, Experiment
from picsellia.types.enums import InferenceType, AnnotationFileType
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from logger import PicselliaLogger
from torch_dataset import HandWrittenCOCODataset

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

cer_metric = evaluate.load("cer")

def compute_cer(pred_ids: torch.Tensor, label_ids: torch.Tensor) -> float:
    """
    Compute CER: it measures the rate of erroneous characters produced by an OCR system compared to the ground truth. It
    is calculated by dividing the total number of incorrect characters by the total number of characters in the
    reference text. CER is expressed as a percentage.
    :param pred_ids: Predictions done by the model;
    :param label_ids: ground-truth
    :return: CER metric
    """
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

def get_encoder_decoder_model(model_weights: str) -> torch.nn:
    """
    Configure vision encoder decoder model.
    :param model_weights: model weights
    :return: configured model
    """
    model = VisionEncoderDecoderModel.from_pretrained(model_weights)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 10
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model

def train_model(model: torch.nn.Module, nb_epochs: int, picsellia_logger: PicselliaLogger, max_length_characters: int) -> torch.nn.Module:
    """
    Train model function
    :param model: model which will be trained in the function
    :param nb_epochs: number of epochs to train the model
    :param picsellia_logger: class which logs the training metrics into Picsell.ia platform.
    :return: Trained model.
    """
    picsellia_logger.on_train_begin()

    for epoch in range(nb_epochs):  # loop over the dataset multiple times
        # train
        model.train()
        train_loss = 0.0

        with tqdm(train_dataloader, unit="batch") as t_epoch:
            for batch in t_epoch:
                t_epoch.set_description(f"Epoch {epoch}")

                # get the inputs
                for k, v in batch.items():
                    batch[k] = v.to(device)

                # forward + backward + optimize
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()

        # evaluate
        model.eval()
        valid_cer = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device), max_length=max_length_characters)
                # compute metrics
                cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                valid_cer += cer

        t_epoch.set_postfix(
            training_loss=train_loss / len(train_dataloader),
            validation_CER=valid_cer / len(val_dataloader))

        picsellia_logger.on_epoch_end(epoch=epoch,
                                      train_loss=train_loss / len(train_dataloader),
                                      val_cer=valid_cer / len(val_dataloader),
                                      display_gpu_occupancy=True if torch.cuda.is_available() else False)

    return model

def transform_bbox_text2label(asset: Asset, classification_dataset_version: DatasetVersion) -> None:
    """
    Extract text from the first bounding box of an asset (from object detection dataset) and create a label in
    classification dataset (containing the same pictures) with the extracted text.
    :param asset: object detection asset
    :param classification_dataset_version: classification dataset version
    """
    filename = asset.filename
    try:
        ann = asset.get_annotation()
        rectangles = ann.list_rectangles()

        if rectangles != []:
            rectangle = rectangles[0]
            text = rectangle.text
            asset_classification = classification_dataset_version.find_asset(filename=filename)

            if text != '':
                label = classification_dataset_version.get_or_create_label(name=text)
            else:
                label = classification_dataset_version.get_or_create_label(name='-')

            classification_annotation = asset_classification.create_annotation()
            classification_annotation.create_classification(label=label)

    except picsellia.exceptions.ResourceNotFoundError:
        logging.warning(f'Not any annotation found in asset {filename}')

    except picsellia.exceptions.ResourceConflictError:
        logging.warning(f'Asset {filename} was already annotated')

    picsellia.exceptions.BadRequestError

    except TypeError:
        logging.warning(f'Impossible to create label {text}')



def fill_picsellia_evaluation_tab(model: torch.nn.Module, data_loader: DataLoader, test_dataset_version_id: UUID,
                                  max_length_characters: int, experiment: Experiment) -> None:
    """
    Fill Picsellia evaluation which allows comparing on a dedicated bench of images the prediction done by the freshly
    trained model with the ground-truth.
    :param model: Model which will be used to do the predictions
    :param data_loader: Dataloader which gathers the bench of images on which the evaluation will be done
    :param test_dataset_version_id: ID of the dataset version which comports the pictures on which the evaluation will be done
    """

    # todo fork classification dataset and infer on it
    object_detection_dataset_version = client.get_dataset_version_by_id(test_dataset_version_id)

    new_dataset_classification_version, _ = object_detection_dataset_version.fork(
        version=object_detection_dataset_version.version + "_classification",
        description="Dataset version created by via experiment trOCR",
        type=InferenceType.CLASSIFICATION
    )

    Parallel(n_jobs=os.cpu_count())(delayed(transform_bbox_text2label)(asset, new_dataset_classification_version) for asset in object_detection_dataset_version.list_assets())

    experiment.attach_dataset(name=f'{object_detection_dataset_version.version}_classification',
                              dataset_version=new_dataset_classification_version)

    model.eval()

    for batch in tqdm(data_loader):

        with torch.no_grad():
            outputs = model.generate(batch["pixel_values"].to(device), max_length=max_length_characters)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)

        for filename, text in zip(batch['filename'], generated_text):
            try:
                asset = new_dataset_classification_version.find_asset(filename=filename)

                label = new_dataset_classification_version.get_or_create_label(name=text)
                experiment.add_evaluation(asset, classifications=[(label, 1.0)])

            except picsellia.exceptions.ResourceNotFoundError:
                logging.warning(f'Filename {filename} was not found in dataset_id: {test_dataset_version_id}')

            except TypeError:
                logging.warning(f'Impossible to create label {text}')

    job = experiment.compute_evaluations_metrics(InferenceType.CLASSIFICATION)
    job.wait_for_done()

def download_coco_file(dataset_path: str, dataset_version: DatasetVersion) -> None:
    coco_filepath = dataset_version.export_annotation_file(annotation_file_type=AnnotationFileType.COCO,
                                                           target_path=os.path.join(dataset_path, 'tmp'))
    os.rename(coco_filepath, os.path.join(dataset_path, 'COCO.json'))
    shutil.rmtree(os.path.join(dataset_path, 'tmp'))


if __name__ == '__main__':
    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]

    client = Client(api_token=api_token, organization_id=organization_id)
    experiment = client.get_experiment_by_id(id=os.environ["experiment_id"])

    context = experiment.get_log(name='parameters').data
    nb_epochs  = context["nb_epochs"]
    learning_rate = context["learning_rate"]
    test_size = context["test_size"]
    batch_size = context["batch_size"]

    if 'max_length_characters' in context.keys():
        max_length_characters = context["max_length_characters"]
    else:
        max_length_characters = 5
        logging.info(f'max_length_characters not found: {max_length_characters} is set by default.')

    logging.info('Parameters used for the training: ')
    for k, v in context.items():
        logging.info(f'     - {k}: {v}')

    # val_object_detection_dataset = client.get_dataset_version_by_id(eval(context["val_object_detection_id"]))
    # train_object_detection_dataset = client.get_dataset_version_by_id(eval(context["train_object_detection_id"]))
    pretrained_trOCR_processor_weights = context["pretrained_trOCRprocessor"]
    train_dataset_version_id = experiment.get_dataset(name='train').id
    val_dataset_version_id = experiment.get_dataset(name='val').id

    if len(experiment.list_attached_dataset_versions()) == 3:
        test_dataset_version_id = experiment.get_dataset(name='test').id
    else:
        test_dataset_version_id = experiment.get_dataset(name='val').id


    vision_encoder_decoder_model_weights = context["vision_encoder_decoder_model_weights"]
    num_workers: Optional[int] = context["num_workers"] if context["num_workers"] is not None else os.cpu_count()


    root_dataset_path = '../../datasets'
    dataset_train_path = os.path.join(root_dataset_path, 'train')
    dataset_validation_path = os.path.join(root_dataset_path, 'val')
    dataset_test_path = os.path.join(root_dataset_path, 'test')



    for dataset_path, dataset_id in zip([dataset_train_path, dataset_validation_path, dataset_test_path],
                                        [train_dataset_version_id, val_dataset_version_id, test_dataset_version_id]) :
        if not os.path.isdir(dataset_path):
            dataset_version = client.get_dataset_version_by_id(dataset_id)
            dataset_version.download(dataset_path)

            # download coco file
            download_coco_file(dataset_path=dataset_path, dataset_version=dataset_version)


    metric = evaluate.load("cer")
    processor = TrOCRProcessor.from_pretrained(pretrained_trOCR_processor_weights)

    train_dataset = HandWrittenCOCODataset(root_dir=dataset_train_path,
                                           coco_filepath=os.path.join(dataset_train_path, 'COCO.json'),
                                           processor=processor,
                                            max_target_length=max_length_characters)
    validation_dataset = HandWrittenCOCODataset(root_dir=dataset_validation_path,
                                                coco_filepath=os.path.join(dataset_validation_path, 'COCO.json'),
                                                processor=processor,
                                                max_target_length=max_length_characters,
                                                is_test=False)
    test_dataset = HandWrittenCOCODataset(root_dir=dataset_test_path,
                                                coco_filepath=os.path.join(dataset_test_path, 'COCO.json'),
                                                processor=processor,
                                                max_target_length=max_length_characters,
                                                is_test=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    picsellia_logger = PicselliaLogger(client=client, experiment=experiment)
    picsellia_logger.log_split_table(
        annotations_in_split={"train": len(train_dataset), "val": len(validation_dataset)},
        title="Nb elem / split")

    model = get_encoder_decoder_model(model_weights=vision_encoder_decoder_model_weights)

    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f'Using device: {device.type.upper()}')
    model.to(device)


    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model = train_model(model=model, nb_epochs=nb_epochs, picsellia_logger=picsellia_logger,
                        max_length_characters=max_length_characters)

    # save model
    model_path = 'model'

    model.save_pretrained(model_path)
    #picsellia_logger.store_model(model_path=model_path, model_name='model-best')


    # Evaluate
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    fill_picsellia_evaluation_tab(model=model,
                                  data_loader=test_data_loader,
                                  test_dataset_version_id=val_dataset_version_id,
                                  max_length_characters=max_length_characters,
                                  experiment=experiment)
