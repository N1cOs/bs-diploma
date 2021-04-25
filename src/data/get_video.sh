#!/bin/bash

set -e

if [[ -n $2 ]]; then
  cd "$2"
fi

case $1 in
  hd-15s-30fps)
    wget 'https://drive.google.com/uc?id=18GOLI8xSDIv_RsHDOChobsh3YF1Pl5EP&export=download' -O "$1".mp4
    ;;
  hd-60s-30fps)
    wget 'https://drive.google.com/uc?id=1VbeRKJHbsUVbFcT9Rzz0CuqFvYZD5RBd&export=download' -O "$1".mp4
    ;;
  *)
    echo "Unknown video"
    exit 1
    ;;
esac
