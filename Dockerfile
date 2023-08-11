FROM nvidia/cuda:11.3.1-devel-ubuntu18.04
LABEL org.iqtlabs.name Anomaly Detector

RUN apt update && apt install -y \
    zlib1g-dev \
    libjpeg-dev \
    python3 \
    python3-pip \
    emacs \
    less \
    tree \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install \
    torch \
    torchvision \
    scikit-learn \
    tqdm

WORKDIR /root
COPY anomaly.py ./
COPY example_program.py ./
COPY example_dataset ./example_dataset
