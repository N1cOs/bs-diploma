import argparse
import glob
import time
from os import path

import cv2

from client import video
from worker import detector

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

    detector = detector.DarknetObjectDetector(args.config, args.weights)

    classes = []
    with open(args.classes, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                classes.append(line)
    writer = video.DetectionWriter(classes)

    times = []
    for img_path in glob.glob(path.join(args.images_dir, "*.jpg")):
        img = cv2.imread(img_path)
        start_time = time.perf_counter()
        detections = detector.detect(img)
        elapsed = round(time.perf_counter() - start_time, 2)

        times.append(elapsed)
        print(f"{img_path}: inference time: {elapsed}s")

        for detection in detections:
            writer.write(img, detection)

        img_path, ext = path.splitext(img_path)
        cv2.imwrite(f"{img_path}.out{ext}", img)

    median = sorted(times)[len(times) // 2]
    print(f"median inference time: {median}s")
