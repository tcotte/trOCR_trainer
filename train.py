import os

import evaluate
import pandas as pd
import torch
from picsellia import Client
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from torch_dataset import IAMDataset

cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer


if __name__ == '__main__':
    if not os.path.isdir('data'):
        client = Client(os.getenv('PICSELLIA_TOKEN'), organization_name=os.getenv('ORGANIZATION_NAME'))
        dataset_version = client.get_dataset_version_by_id('019585ab-3cb0-7b59-9f6f-b74818844d2b')
        dataset_version.download('data')

    metric = evaluate.load("cer")
    # dataset = HandWrittenDataset(root_dir='data', csv_path='crop_data.csv')
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    df = pd.read_csv('crop_data.csv')
    df = df[~df['number'].isnull()]

    train_df, test_df = train_test_split(df, test_size=0.2)
    print(len(train_df), len(test_df))


    train_dataset = IAMDataset(root_dir='data', df=train_df, processor=processor)
    test_dataset = IAMDataset(root_dir='data', df=test_df, processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=4)

    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

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



    # training_args = Seq2SeqTrainingArguments(
    #     num_train_epochs=3,
    #     predict_with_generate=True,
    #     evaluation_strategy="steps",
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     fp16=True,
    #     output_dir=".",
    #     logging_steps=2,
    #     save_steps=1000,
    #     eval_steps=200,
    #     save_total_limit=1,
    # )
    #
    # # instantiate trainer
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     tokenizer=processor.feature_extractor,
    #     args=training_args,
    #     compute_metrics=compute_metrics,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     data_collator=default_data_collator,
    # )
    #
    # trainer.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(10):  # loop over the dataset multiple times
        # train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
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

        print(f"Loss after epoch {epoch}:", train_loss / len(train_dataloader))

        # evaluate
        model.eval()
        valid_cer = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device))
                # compute metrics
                cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                valid_cer += cer

        print("Validation CER:", valid_cer / len(eval_dataloader))

    model.save_pretrained(".")

    os.makedirs("model/")
    model.save_pretrained("model/")
