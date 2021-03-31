FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install --yes \
    libgomp1

ARG REQUIREMENTS

COPY $REQUIREMENTS .

RUN pip install -r $REQUIREMENTS

COPY . .
