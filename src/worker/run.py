import argparse
import logging
import sys
import time

import detector
import zmq

import proto

LOGGER_NAME = "worker"
LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

if __name__ == "__main__":
    targets = detector.get_available_targets()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ventilator-addr", required=True)
    parser.add_argument("--collector-addr", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--target", choices=list(targets), default="cpu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    log = logging.getLogger(LOGGER_NAME)
    logging.basicConfig(
        level=args.log_level.upper(),
        format=LOG_FORMAT,
        stream=sys.stdout,
    )
    log.info(f"successfully started: args={args}")

    detector = detector.DarknetObjectDetector(
        args.config,
        args.weights,
        targets[args.target],
    )
    log.info("successfully loaded object detector")

    context = zmq.Context()
    vent_socket = context.socket(zmq.PULL)
    vent_socket.connect(f"tcp://{args.ventilator_addr}")

    collect_socket = context.socket(zmq.PUSH)
    collect_socket.connect(f"tcp://{args.collector_addr}")

    while True:
        msg = vent_socket.recv(copy=False)
        req = proto.parse_detect_request(msg)
        log.debug(f"got detect request: id={req.id}")

        start = time.perf_counter()
        detections = detector.detect(req.img)

        elapsed = time.perf_counter() - start
        log.debug(f"processed image: id={req.id}, elapsed={round(elapsed, 2)}s")

        resp = proto.DetectResponse(req.id, elapsed, detections)
        collect_socket.send(proto.dump_detect_response(resp))
