#!/bin/bash

set -e

if [[ -n $2 ]]; then
  cd "$2"
fi

case $1 in
  yolov4-tiny)
    wget 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'
    wget 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights'
    ;;
  yolov4-416)
    wget 'https://drive.google.com/uc?id=1Ahmn7iI2B79jDlcKlRlD_tO5Tbr0Dq8x&export=download' -O "$1".cfg
    wget 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights'
    ;;
  yolov3)
    wget 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
    wget 'https://pjreddie.com/media/files/yolov3.weights'
    ;;
  *)
    echo "Unknown model name"
    exit 1
    ;;
esac
