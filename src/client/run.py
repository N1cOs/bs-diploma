import argparse
import asyncio
import logging
import os
import sys
import time

import collector
import zmq
import zmq.asyncio

import proto
import video

LOGGER_NAME = "client"
LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", required=True)
    parser.add_argument("--in-video", required=True)
    parser.add_argument("--out-video", default="")
    parser.add_argument("--write-label", type=bool, default=True)
    parser.add_argument("--read-buf-size", type=int, default=30)
    parser.add_argument("--write-buf-size", type=int, default=60)
    parser.add_argument("--frame-timeout-sec", type=int, default=5)
    parser.add_argument("--ventilator-addr", default="0.0.0.0:5000")
    parser.add_argument("--collector-addr", default="0.0.0.0:5001")
    # ToDo: replace to aiohttp: need to set up monitoring services
    parser.add_argument("--startup-wait-sec", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-progress-percent", type=int, default=10)
    args = parser.parse_args()

    if not args.out_video:
        path, ext = os.path.splitext(args.in_video)
        args.out_video = f"{path}.out{ext}"

    log = logging.getLogger(LOGGER_NAME)
    logging.basicConfig(
        level=args.log_level.upper(),
        format=LOG_FORMAT,
        stream=sys.stdout,
    )
    log.info(f"started: args={args}")
    time.sleep(args.startup_wait_sec)

    classes = []
    with open(args.classes, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                classes.append(line)

    frames = collector.AsyncDict(args.read_buf_size)
    frame_reader = video.AsyncFrameReader(args.in_video)
    video_stat = frame_reader.video_stat
    frame_writer = video.AsyncFrameWriter(args.out_video, "mp4v", video_stat)
    log.info(f"input video: {video_stat}")

    ctx = zmq.asyncio.Context()
    collect_sock = ctx.socket(zmq.PULL)
    collect_sock.bind(f"tcp://{args.collector_addr}")

    writer = collector.DetectionCollector(
        sock=collect_sock,
        max_rcv_buf=args.write_buf_size,
        frame_timeout=args.frame_timeout_sec,
        classes=classes,
        write_label=args.write_label,
        frame_dict=frames,
        frame_writer=frame_writer,
    )

    vent_sock = ctx.socket(zmq.PUSH)
    vent_sock.bind(f"tcp://{args.ventilator_addr}")

    # ToDo: handle id overflowing because it's 4 byte size in message
    id_ = 0
    start = time.perf_counter()
    async with frame_reader as reader:
        has_frame, frame = await reader.read()
        if not has_frame:
            return

        await frames.put(id_, frame)
        await send_frame(vent_sock, id_, frame, log)
        write_task = writer.start()

        log_step = args.log_progress_percent / 100
        next_log = log_step
        while not reader.closed:
            id_ += 1
            if (sent := id_ / video_stat.frames) >= next_log:
                sent = round(sent * 100, 2)
                elapsed = round(time.perf_counter() - start, 2)
                log.info(f"progress: sent={sent}%,elapsed={elapsed}s")
                next_log += log_step

            has_frame, frame = await reader.read()
            if not has_frame:
                break

            await frames.put(id_, frame)
            await send_frame(vent_sock, id_, frame, log)

    await write_task

    elapsed = round(time.perf_counter() - start, 2)
    log.info(
        f"processed whole video: elapsed={elapsed}s, dropped frames={writer.dropped_frames}"
    )


async def send_frame(sock, id_, frame, log):
    req = proto.DetectRequest(id_, frame)
    data = proto.dump_detect_request(req)
    await sock.send(data, copy=False)
    log.debug(f"sent frame: id={id_}")


if __name__ == "__main__":
    asyncio.run(main())
