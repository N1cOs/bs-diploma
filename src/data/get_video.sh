#!/bin/bash

set -e

if [[ -n $2 ]]; then
  cd "$2"
fi

case $1 in
  hd-15s-30fps)
    wget 'https://drive.google.com/uc?id=1rfb91WT3NYUHXiRafvrjYbLijMPf3xo5&export=download' -O "$1".mp4
    ;;
  full-15s-30fps)
    wget 'https://drive.google.com/uc?id=1_DHMYTwblz8VIvYL-K_7tWYKGiM7uaEa&export=download' -O "$1".mp4
    ;;
  hd-30s-30fps)
    wget 'https://drive.google.com/uc?id=1SQ076jjI_nC7zR3V0r9D02X-OXxgCf8s&export=download' -O "$1".mp4
    ;;
  full-30s-30fps)
    wget 'https://drive.google.com/uc?id=19fcvpjWZ-b5vogd_ou41kaVSioWADgz4&export=download' -O "$1".mp4
    ;;
  *)
    echo "Unknown video"
    exit 1
    ;;
esac
