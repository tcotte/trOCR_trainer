FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.9
LABEL authors="Tristan_COTTE"

RUN pip3 config set global.trusted-host "download.pytorch.org cdn-lfs.hf.co huggingface.co"

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install evaluate
RUN pip3 install jiwer
RUN pip3 install picsellia
RUN pip3 install scikit-learn
RUN pip3 install transformers

WORKDIR /picsellia_folder


COPY picsellia_folder .

RUN mkdir -p /saved_models && chmod 777 /saved_models
RUN mkdir -p /datasets && chmod 777 /datasets
RUN mkdir -p /workspace && chmod 777 /workspace
RUN chmod 777 ./

ENTRYPOINT ["python3.9", "train.py"]