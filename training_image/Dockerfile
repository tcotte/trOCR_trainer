FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.9
LABEL authors="Tristan_COTTE"

RUN pip3 config set global.trusted-host "download.pytorch.org cdn-lfs.hf.co huggingface.co"

RUN pip3 install evaluate
RUN pip3 install jiwer
RUN pip3 install picsellia

WORKDIR /picsellia_folder


COPY training_image/picsellia_folder .

RUN mkdir -p /saved_models && chmod 777 /saved_models
RUN mkdir -p /datasets && chmod 777 /datasets
RUN chmod 777 ./

ENTRYPOINT ["python3.9", "train.py"]