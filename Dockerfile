FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    # need to build numpy
    gcc \
    git \
    # need for torch
    libopenblas-dev \
    # need for torchvision
    libjpeg-dev \
    zlib1g-dev

# need main because build fails on older versions
ARG NPY_REF=main

RUN git clone --quiet --depth 1 --branch ${NPY_REF} https://github.com/numpy/numpy.git && \
    cd numpy && \
    pip install .

COPY . .

RUN pip install wheels/*.whl
