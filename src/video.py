import argparse
import dataclasses
import datetime
import sys
from os import path

import cv2

from lib import darknet


@dataclasses.dataclass
class VideoStat:
    width: int
    height: int
    fps: float
    frames: int
    duration: float

    @classmethod
    def from_video_capture(cls, cap: cv2.VideoCapture):
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cls(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frames=frames,
            duration=frames / fps,
        )

    def __str__(self):
        return (
            f"video stats: resolution={self.width}x{self.height}, "
            f"duration={self.duration:.2f}s, fps={self.fps:.2f}"
        )


def print_progress(frames: int, all_frames: int, start: datetime.datetime):
    percent = round((frames / all_frames) * 100, 2)
    elapsed = datetime.datetime.now() - start
    info = f"\rprogress: processed={percent}%, elapsed={elapsed.seconds}s"
    sys.stdout.write(info)
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="data/yolov3.cfg")
    parser.add_argument("-w", "--weights", default="data/yolov3.weights")
    parser.add_argument("-l", "--classes", default="data/coco.names")
    parser.add_argument("-i", "--in-video", default="data/video.mp4")
    parser.add_argument("-o", "--out-video", default="")
    args = parser.parse_args()

    if not args.out_video:
        path, ext = path.splitext(args.in_video)
        args.out_video = f"{path}.out{ext}"

    print(f"started with args: {args}")

    net = darknet.Network.from_config(args.config)
    net.load_weights(args.weights)

    classes = []
    with open(args.classes, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                classes.append(line)
    detector = darknet.ObjectDetector(net)

    cap = cv2.VideoCapture(args.in_video)
    stats = VideoStat.from_video_capture(cap)
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
                p1 = (int(detection.x1), int(detection.y1))
                p2 = (int(detection.x2), int(detection.y2))
                # ToDo: use separate color for each class
                color = (255, 0, 0)
                cv2.rectangle(frame, p1, p2, color)

            writer.write(frame)
            frames += 1
    finally:
        cap.release()
        writer.release()

        print_progress(frames, stats.frames, start_time)
        sys.stdout.write("\n")
