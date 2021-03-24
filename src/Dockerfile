FROM python:3.8-slim

WORKDIR /app

COPY requirements-arm64.txt .

RUN pip install -r requirements-arm64.txt

COPY . .
