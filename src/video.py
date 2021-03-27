import argparse
import datetime
import sys
from os import path

import cv2

from lib import darknet, view, video


def print_progress(frames: int, all_frames: int, start: datetime.datetime):
    percent = round((frames / all_frames) * 100, 2)
    elapsed = datetime.datetime.now() - start
    info = f"\rprogress: processed={percent}%, elapsed={elapsed.seconds}s"
    sys.stdout.write(info)
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=path.join("data", "yolov3.cfg"))
    parser.add_argument("-w", "--weights", default=path.join("data", "yolov3.weights"))
    parser.add_argument("-l", "--classes", default=path.join("data", "coco.names"))
    parser.add_argument("-i", "--in-video", default=path.join("data", "video_15s.mp4"))
    parser.add_argument("-o", "--out-video", default="")
    args = parser.parse_args()

    if not args.out_video:
        path, ext = path.splitext(args.in_video)
        args.out_video = f"{path}.out{ext}"

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
    detection_writer = view.DetectionWriter(classes)

    cap = cv2.VideoCapture(args.in_video)
    stats = video.VideoStat.from_video_capture(cap)
    print(stats)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        args.out_video, fourcc, stats.fps, (stats.width, stats.height)
    )

    frames = 0
    start_time = datetime.datetime.now()
    try:
        while cap.isOpened():
            has_frame, frame = cap.read()
            if not has_frame:
                break
            print_progress(frames, stats.frames, start_time)

            detections = detector.detect(frame)
            for detection in detections:
                detection_writer.write_detection(frame, detection)

            writer.write(frame)
            frames += 1
    finally:
        cap.release()
        writer.release()

        print_progress(frames, stats.frames, start_time)
        sys.stdout.write("\n")
