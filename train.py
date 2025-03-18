import logging
import os
from typing import Optional

import evaluate
import pandas as pd
import torch
from picsellia import Client
from picsellia.types.enums import InferenceType

from logger import PicselliaLogger
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from prepare_dataset import create_dataframe_with_bboxes
from torch_dataset import IAMDataset, HandWrittenTestDataset

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

cer_metric = evaluate.load("cer")

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

def get_encoder_decoder_model(model_weights: str) -> torch.nn:
    model = VisionEncoderDecoderModel.from_pretrained(vision_encoder_decoder_model_weights)

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

def train_model(model: torch.nn.Module, nb_epochs: int, picsellia_logger: PicselliaLogger):
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
            for batch in eval_dataloader:
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device))
                # compute metrics
                cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                valid_cer += cer

        t_epoch.set_postfix(
            training_loss=train_loss / len(train_dataloader),
            validation_CER=valid_cer / len(eval_dataloader))

        picsellia_logger.on_epoch_end(epoch=epoch,
                                      train_loss=train_loss / len(train_dataloader),
                                      val_cer=valid_cer / len(eval_dataloader),
                                      display_gpu_occupancy=True if torch.cuda.is_available() else False)

    return model

def fill_picsellia_evaluation_tab(model: torch.nn.Module, data_loader) -> None:
    model.eval()

    for batch in tqdm(data_loader):

        with torch.no_grad():
            outputs = model.generate(batch["pixel_values"].to(device))
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)

        for filename, text in zip(batch['filename'], generated_text):
            asset = val_object_detection_dataset.find_asset(filename=filename)
            experiment.add_evaluation(asset, classifications=text)
            job = experiment.compute_evaluations_metrics(InferenceType.OBJECT_DETECTION)

    job.wait_for_done()

if __name__ == '__main__':
    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]
    # job_id = os.environ["job_id"]

    client = Client(api_token=api_token, organization_id=organization_id)
    experiment = client.get_experiment_by_id(id=os.environ["experiment_id"])

    context = experiment.get_log(name='parameters').data
    nb_epochs  = context["nb_epochs"]
    learning_rate = context["learning_rate"]
    test_size = context["test_size"]
    batch_size = context["batch_size"]
    val_object_detection_dataset = client.get_dataset_version_by_id(eval(context["val_object_detection_id"]))
    train_object_detection_dataset = client.get_dataset_version_by_id(eval(context["train_object_detection_id"]))
    pretrained_trOCR_processor_weights = context["pretrained_trOCRprocessor"]
    train_classification_dataset_id = experiment.get_dataset(name='train').id
    vision_encoder_decoder_model_weights = context["vision_encoder_decoder_model_weights"]
    num_workers: Optional[int] = context["num_workers"] if context["num_workers"] is not None else os.cpu_count()



    root_dataset_path = 'datasets'
    dataset_train_path = os.path.join(root_dataset_path, 'train_data')
    dataset_test_path = os.path.join(root_dataset_path, 'test_data')

    for dataset_path, dataset_id in zip([dataset_train_path, dataset_test_path],
                                [train_classification_dataset_id, eval(context["val_object_detection_id"])]) :
        if not os.path.isdir(dataset_path):
            dataset_version = client.get_dataset_version_by_id(dataset_id)
            dataset_version.download(dataset_path)

    metric = evaluate.load("cer")
    processor = TrOCRProcessor.from_pretrained(pretrained_trOCR_processor_weights)

    if not os.path.exists(os.path.join(root_dataset_path, 'data.csv')):
        df = create_dataframe_with_bboxes(dataset_object_detection_version=train_object_detection_dataset,
            classification_dataset_versions=client.get_dataset_version_by_id(train_classification_dataset_id))
        df = df[~df['number'].isnull()]

        # temporary
        df.to_csv(os.path.join(root_dataset_path, 'data.csv'))

    else:
        df = pd.read_csv(os.path.join(root_dataset_path, 'data.csv'))

    train_df, validation_df = train_test_split(df, test_size=test_size, random_state=42)

    picsellia_logger = PicselliaLogger(client=client, experiment=experiment)
    picsellia_logger.log_split_table(
        annotations_in_split={"train": len(train_df), "val": len(validation_df)},
        title="Nb elem / split")


    train_dataset = IAMDataset(root_dir=dataset_train_path, df=train_df, processor=processor)
    validation_dataset = IAMDataset(root_dir=dataset_train_path, df=validation_df, processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = get_encoder_decoder_model(model_weights=vision_encoder_decoder_model_weights)

    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f'Using device: {device.type.upper()}')
    model.to(device)


    optimizer = AdamW(model.parameters(), lr=learning_rate)

    picsellia_logger.on_train_begin()

    model = train_model(model=model, nb_epochs=nb_epochs, picsellia_logger=picsellia_logger)

    # save model
    model.save_pretrained(".")

    model_path = 'saved_models'
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(model_path, 'best.pth')
    torch.save(model.state_dict(), model_path)

    picsellia_logger.store_model(model_path=model_path, model_name='model-best')
    # model.save_pretrained("model/")

    '''
    # TODO try to load model like this (it will save space on disk):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch 
    
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    model.load_state_dict(torch.load('/content/trOCR_trainer/finetuned_model_weights.pth'))
    '''
    # Evaluate
    test_dataset = HandWrittenTestDataset(root_dir=dataset_test_path,
                                          object_detection_dataset_version=val_object_detection_dataset,
                                          processor=processor)

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    fill_picsellia_evaluation_tab(model=model, data_loader=test_data_loader)
