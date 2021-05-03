import asyncio
import dataclasses
import heapq
import logging
import time
from typing import List

import numpy as np
import zmq

import proto
import stats
import video


class DetectionCollector:
    def __init__(
        self,
        sock: zmq.Socket,
        recv_buf_size: int,
        write_buf_size: int,
        frame_timeout: int,
        classes: List[str],
        write_label: bool,
        frame_dict: "AsyncDict",
        frame_writer: video.AsyncFrameWriter,
        stats_ctl: stats.FrameEventsCollector,
    ):
        self.sock = sock
        self.recv_buf_size = recv_buf_size
        self.frame_timeout = frame_timeout

        self.detection_writer = video.AsyncDetectionWriter(classes, write_label)
        self.frame_dict = frame_dict
        self.frame_writer = frame_writer
        self.stats_ctl = stats_ctl

        self._dropped = 0
        self._write_queue = asyncio.Queue(maxsize=write_buf_size)
        self._stopping = asyncio.Event()
        self._log = logging.getLogger(__name__)

    @property
    def dropped_frames(self):
        return self._dropped

    def start(self) -> asyncio.Task:
        write_task = asyncio.create_task(self._write_detections(), name="write_task")
        return asyncio.create_task(self._poll_detections(write_task), name="poll_task")

    async def _poll_detections(self, write_task: asyncio.Task):
        name = asyncio.current_task().get_name()
        self._log.info(f"{name}: started")
        start_time = time.perf_counter()

        pq = []
        frames = {}

        async def write_head():
            head = heapq.heappop(pq)
            frame = frames.pop(head.id)
            self._log.debug(f"{name}: writing frame: id={head.id}")
            await self._write_queue.put((frame.data, head.detections))
            return head.id

        last_written = -1
        while True:
            try:
                raw_resp = await asyncio.wait_for(self.sock.recv(), self.frame_timeout)
                resp = proto.parse_detect_response(raw_resp)
                frame = await self.frame_dict.pop(resp.id)
                self._log.debug(f"{name}: received frame: id={resp.id}")
                await self.stats_ctl.send_transfer_duration(
                    time.perf_counter() - frame.sent_time - resp.elapsed_sec,
                )
                await self.stats_ctl.send_detect_duration(resp.elapsed_sec)

                if resp.id <= last_written:
                    self._log.warning(f"{name}: dropping frame: id={resp.id}")
                    self._dropped += 1
                    continue

                heapq.heappush(pq, resp)
                frames[resp.id] = frame

                while pq and pq[0].id - 1 == last_written:
                    await write_head()
                    last_written += 1

                if len(pq) > self.recv_buf_size:
                    # write nearest frame which we already received
                    id_ = await write_head()
                    last_written = id_
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - start_time
                elapsed = round(elapsed - self.frame_timeout, 2)
                self._log.info(
                    f"{name}: got frame timeout, stopping: processing time={elapsed}s"
                )
                self._stopping.set()
                await write_task
                break
            except asyncio.CancelledError:
                self._log.info(f"{name}: got cancel signal, stopping")
                break
            except:
                self._log.error(f"{name}: unhandled exception", exc_info=True)

        self._log.info(f"{name}: finished")

    async def _write_detections(self):
        name = asyncio.current_task().get_name()
        self._log.info(f"{name}: started")

        async def write(res):
            frame, detections = res
            now = time.perf_counter()
            for d in detections:
                await self.detection_writer.write(frame, d)
            await self.stats_ctl.send_write_detections_duration(
                time.perf_counter() - now
            )

            now = time.perf_counter()
            await self.frame_writer.write(frame)
            await self.stats_ctl.send_encode_duration(time.perf_counter() - now)

        async def write_until_empty(q):
            while not q.empty():
                res = await q.get()
                await write(res)

        wait_task = asyncio.create_task(self._stopping.wait())
        async with self.frame_writer:
            while True:
                try:
                    write_task = asyncio.create_task(self._write_queue.get())
                    tasks = {wait_task, write_task}

                    done, _ = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    if write_task in done:
                        res = write_task.result()
                        await write(res)
                    elif wait_task in done:
                        self._log.info(f"{name}: got stopping signal")
                        await write_until_empty(self._write_queue)
                        break
                    else:
                        raise ValueError(f"unknown done tasks: {done}")
                except asyncio.CancelledError:
                    self._log.info(
                        f"{name}: got cancel signal, writing all buffered detections"
                    )
                    await write_until_empty(self._write_queue)
                    break
                except:
                    self._log.error(f"{name}: unhandled exception", exc_info=True)

        self._log.info(f"{name}: finished")


@dataclasses.dataclass
class Frame:
    data: np.ndarray
    sent_time: float


class AsyncDict:
    def __init__(self, maxsize: int):
        self._dict = {}
        self._queue = asyncio.Queue(maxsize=maxsize)

    async def put(self, key, val):
        await self._queue.put(True)
        self._dict[key] = val

    async def pop(self, key):
        await self._queue.get()
        return self._dict.pop(key)
