FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.9
LABEL authors="Tristan_COTTE"

# Enable permission to downlaod from certain sites
RUN pip3 config set global.trusted-host "download.pytorch.org cdn-lfs.hf.co huggingface.co"

RUN pip3 install python-dotenv
RUN pip3 install picsellia
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install opencv-python-headless
RUN pip3 install transformers

RUN pip3 install protobuf
RUN pip3 install --upgrade protobuf==3.20.1
RUN pip3 install pandas
RUN pip3 install joblib


COPY . .

RUN mkdir /workspace
RUN chown -R 42420 ./workspace

CMD ["trOCR_pre_annotation/launch_processing.py"]
ENTRYPOINT ["python3.9"]
RUN chown -R 42420 ./trOCR_pre_annotation
