import argparse
import datetime
import glob
from os import path

import cv2

from lib import darknet, view

if __name__ == "__main__":
    """
    Last results of median inference time:
        - YOLOv3-416
            Laptop: > 600ms
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=path.join("data", "yolov3.cfg"))
    parser.add_argument("-w", "--weights", default=path.join("data", "yolov3.weights"))
    parser.add_argument("-l", "--classes", default=path.join("data", "coco.names"))
    parser.add_argument("-i", "--images-dir", default=path.join("data", "images"))

    args = parser.parse_args()
    print(f"started with args: {args}")

    net = darknet.Network.from_config(args.config)
    net.load_weights(args.weights)
    detector = darknet.ObjectDetector(net)

    classes = []
    with open(args.classes, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                classes.append(line)
    writer = view.DetectionWriter(classes)

    times = []
    for img_path in glob.glob(path.join(args.images_dir, "*.jpg")):
        img = cv2.imread(img_path)
        start_time = datetime.datetime.now()
        detections = detector.detect(img)
        elapsed = datetime.datetime.now() - start_time

        times.append(elapsed)
        print(f"{img_path}: inference time: {elapsed.microseconds / 1000}ms")

        for detection in detections:
            writer.write_detection(img, detection)

        img_path, ext = path.splitext(img_path)
        cv2.imwrite(f"{img_path}.out{ext}", img)

    median = sorted(times)[len(times) // 2]
    print(f"median inference time: {median.microseconds / 1000}ms")
