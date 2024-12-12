#FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
#FROM ubuntu18.04

FROM ubuntu:18.04 
ENV NVARCH x86_64

ENV DEBIAN_FRONTEND noninteractive

# Change APT sources to Tsinghua mirror
RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/http:\/\/security.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update --fix-missing

# Sync system time using chrony
RUN apt-get install -y chrony

# Install Python and pip
RUN apt-get install -y python3 python3-pip && \
    python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 5

# Set Tsinghua PyPI mirror
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


RUN pip3 install tensorflow==2.4.0
RUN pip3 install torch==1.7.0
RUN pip3 install tensorflow-datasets

# install openvino
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY
ARG DOWNLOAD_LINK=https://registrationcenter-download.intel.com/akdlm/irc_nas/17662/l_openvino_toolkit_p_2021.3.394.tgz
ARG INSTALL_DIR=/opt/intel/computer_vision_sdk
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN pip3 install networkx
RUN pip3 install defusedxml


# Set locale environment variables for UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# install requirements for yolov5
COPY yolov5-7.0/requirements.txt /yolov5-7.0/requirements.txt
RUN cd /yolov5-7.0 &&  PYTHONIOENCODING=utf-8 pip3 install --no-cache-dir -r requirements.txt
RUN apt-get update --fix-missing
RUN apt-get install -y libgl1 libgtk-3-0 libgtk-3-dev
RUN pip3 install pycocotools

# install requirements for model conversion
RUN pip3 install netron
RUN pip3 install onnx
RUN pip3 install onnx-simplifier
RUN pip3 install tensorflow-datasets
RUN pip3 install openvino2tensorflow==1.17.2
RUN pip3 install gdown

# Add a user that UID:GID will be updated by vscode
ARG USERNAME=developer
ARG GROUPNAME=developer
ARG UID=1000
ARG GID=1000
ARG PASSWORD=developer
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME
ENV HOME /home/developer
