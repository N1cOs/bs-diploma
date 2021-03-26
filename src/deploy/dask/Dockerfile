FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install \
        --yes \
        --no-install-recommends \
            gcc

COPY requirements.txt .

RUN pip install --no-cache-dir --requirement requirements.txt
