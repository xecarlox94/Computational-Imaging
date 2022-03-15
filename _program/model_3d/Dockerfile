FROM nytimes/blender:3.0-gpu-ubuntu18.04

COPY . /src/

VOLUME ./dataset/ /src/dataset

WORKDIR /src/

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install blender python3 python3-pip -y

RUN /bin/bash ./run.sh
#CMD touch /src/dataset/new_file.txt
