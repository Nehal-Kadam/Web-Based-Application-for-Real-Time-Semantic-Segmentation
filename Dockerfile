FROM python:3-slim

ADD https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth /segmentation-service/sam_vit_h_4b8939.pth

RUN apt update && \
    apt install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY . /segmentation-service
WORKDIR /segmentation-service
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "./segment.py"]
