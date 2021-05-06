import asyncio
import dataclasses
import logging


@dataclasses.dataclass
class Event:
    type: int
    seconds: float


class FrameEventsCollector:
    DECODE_EVENT = 0
    DETECT_EVENT = 1
    DETECTION_WRITE_EVENT = 2
    ENCODE_EVENT = 3

    EVENTS_NAME = [
        "decode",
        "detect",
        "detection_write",
        "encode",
    ]

    def __init__(self, enable: bool = False):
        self.enable = enable
        self._queue = asyncio.Queue()
        self._stats = [[] for _ in range(len(self.EVENTS_NAME))]

        self._log = logging.getLogger(__name__)
        if not enable:
            self._log.addHandler(logging.NullHandler())
            self._log.propagate = False

        self._stopping = asyncio.Event()
        self._task = asyncio.create_task(self._read_loop(), name="read_loop")

    async def send_decode_duration(self, seconds: float):
        await self._queue.put(Event(self.DECODE_EVENT, seconds))

    async def send_detect_duration(self, seconds: float):
        await self._queue.put(Event(self.DETECT_EVENT, seconds))

    async def send_write_detections_duration(self, seconds: float):
        await self._queue.put(Event(self.DETECTION_WRITE_EVENT, seconds))

    async def send_encode_duration(self, seconds: float):
        await self._queue.put(Event(self.ENCODE_EVENT, seconds))

    async def stop(self):
        self._stopping.set()
        await self._task

    async def _read_loop(self):
        name = asyncio.current_task().get_name()
        self._log.info(f"{name}: started")

        wait_stop = asyncio.create_task(self._stopping.wait())
        while True:
            try:
                handle_task = asyncio.create_task(self._queue.get())
                tasks = {wait_stop, handle_task}

                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if handle_task in done:
                    event = handle_task.result()
                    if self.enable:
                        self._stats[event.type].append(event.seconds)
                elif wait_stop in done:
                    if self.enable:
                        avg = [sum(s) / len(s) for s in self._stats]
                        all_ = sum(avg)
                        self._log.info(
                            f"{name}: average frame processing time: {round(all_, 2)}s"
                        )
                        for i, v in enumerate(avg):
                            event_name = self.EVENTS_NAME[i]
                            self._log.info(
                                f"{name}: {event_name}: avg={round(v, 3)}s, "
                                f"percentage={round(v/all_ * 100, 2)}%"
                            )
                    break
                else:
                    raise ValueError(f"unknown done tasks: {done}")
            except asyncio.CancelledError:
                self._log.info(f"{name}: got cancel signal")
                break

        self._log.info(f"{name}: finished")
