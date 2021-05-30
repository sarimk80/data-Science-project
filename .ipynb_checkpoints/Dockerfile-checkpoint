# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM ubuntu
#FROM nvcr.io/nvidia/pytorch:21.04-py3
#nvcr.io/nvidia/pytorch:20.03-py3

#FROM ubuntu:20.04
#RUN apt-get update 
#RUN apt-get upgrade -y
#RUN apt-get install -y python3

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get upgrade
#RUN apt-get install -y python3 python3-pip
#RUN apk add pip3

RUN pip3 install --upgrade Pillow
RUN pip3 install gsutil
RUN pip3 install Cython matplotlib numpy opencv-python PyYAML scipy tensorboard tqdm uvicorn fastapi
RUN pip3 install torch
RUN pip3 install torchvision
RUN apt-get update 
#&& apt-get install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get install ffmpeg libsm6 libxext6 libfontconfig1 libxrender1 libgl1-mesa-glx -y
RUN pip3 install ipython
RUN pip3 install -U scikit-learn
RUN pip3 install seaborn
RUN pip3 install python-multipart
RUN pip3 install azure-storage-blob==2.1.0 

# Install dependencies
#COPY requirements.txt .

#RUN pip3 install -r requirements.txt gsutil

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . .
WORKDIR /usr/src/app/master/yolov5

#CMD uvicorn master.yolov5.main:app

#CMD pwd
EXPOSE 80

CMD ["export", "AZURE_STORAGE_CONNECTIONSTRING", "DefaultEndpointsProtocol=https;AccountName=23237iba;AccountKey=0jdjVZ68D9aVSZqcVJOkrvV3X8+/kUZU/Lza3GKqzu4vhOdCuCLkg3SgV8DxEbymI/PrEFwkLXatr2RUg+vxrw==;EndpointSuffix=core.windows.net"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
#CMD ["uvicorn" , "main:app"]

# Copy weights
#RUN python3 -c "from models import *; \
#attempt_download('weights/yolov5s.pt'); \
#attempt_download('weights/yolov5m.pt'); \
#attempt_download('weights/yolov5l.pt')"


# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t

# Kill all
# sudo docker kill "$(sudo docker ps -q)"

# Kill all image-based
# sudo docker kill $(sudo docker ps -a -q --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker container exec -it ba65811811ab bash

# Bash into stopped container
# sudo docker commit 092b16b25c5b usr/resume && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco --entrypoint=sh usr/resume

# Send weights to GCP
# python -c "from utils.general import *; strip_optimizer('runs/exp0_*/weights/last.pt', 'tmp.pt')" && gsutil cp tmp.pt gs://*

# Clean up
# docker system prune -a --volumes
