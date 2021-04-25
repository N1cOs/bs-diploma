import asyncio
import heapq
import logging
from typing import List

import zmq

import proto
import video


class DetectionCollector:
    def __init__(
        self,
        sock: zmq.Socket,
        max_rcv_buf: int,
        frame_timeout: int,
        classes: List[str],
        write_label: bool,
        frame_dict: "AsyncDict",
        frame_writer: video.AsyncFrameWriter,
    ):
        self.sock = sock
        self.max_rcv_buf = max_rcv_buf
        self.frame_timeout = frame_timeout

        self.detection_writer = video.AsyncDetectionWriter(classes, write_label)
        self.frame_dict = frame_dict
        self.frame_writer = frame_writer

        self._dropped = 0
        self._write_queue = asyncio.Queue()
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

        pq = []
        frames = {}

        async def write_head():
            head = heapq.heappop(pq)
            frame = frames.pop(head.id)
            self._log.debug(f"{name}: writing frame: id={head.id}")
            await self._write_queue.put((frame, head.detections))
            return head.id

        last_written = -1
        while True:
            try:
                raw_resp = await asyncio.wait_for(self.sock.recv(), self.frame_timeout)
                resp = proto.parse_detect_response(raw_resp)
                frame = await self.frame_dict.pop(resp.id)
                self._log.debug(f"{name}: received frame: id={resp.id}")

                if resp.id <= last_written:
                    self._log.warning(f"{name}: dropping frame: id={resp.id}")
                    self._dropped += 1
                    continue

                heapq.heappush(pq, resp)
                frames[resp.id] = frame

                while pq and pq[0].id - 1 == last_written:
                    await write_head()
                    last_written += 1

                if len(pq) > self.max_rcv_buf:
                    # write nearest frame which we already received
                    id_ = await write_head()
                    last_written = id_
            except asyncio.TimeoutError:
                self._log.info(f"{name}: got frame timeout, stopping")
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
            for d in detections:
                await self.detection_writer.write(frame, d)
            await self.frame_writer.write(frame)

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
