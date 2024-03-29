# Bachelor's Thesis

## Description
Repository contains my bachelor's thesis at ITMO University. The paper was about development of a distributed algorithm for object detection in videos. Used technologies:
* asyncio;
* OpenCV (with Tengine);
* ZeroMQ.

https://user-images.githubusercontent.com/38363125/120944162-ad55a080-c73b-11eb-9120-7774d1460842.mp4

## Deployment
Developed algorithm was tested on a cluster with 22 single-board computers [ROCKPro64](https://wiki.pine64.org/wiki/ROCKPro64). To ease deployment process, Docker was installed on each node and then all nodes were joined in Docker Swarm.
Complete configs for deployment can be find in the [src/deploy](src/deploy) directory. A deployment diagram:

![Deployment diagram](report/images/deployment_dgrm.png)

## Results
Two object detection models were used for testing: yolov4-416 and yolov4-tiny. More information about these models you can find in Darknet [repository](https://github.com/AlexeyAB/darknet). Results for yolov4-tiny:

![yolo_tiny_results](https://user-images.githubusercontent.com/38363125/121081508-7cca4100-c7e5-11eb-8a1a-aa00248b1b48.png)

Results for yolov4-416:

![yolo_416_results)](https://user-images.githubusercontent.com/38363125/121081570-8fdd1100-c7e5-11eb-8d81-556082d39e11.png)

Comparison of a distributed solution and processing video on a single node:
| Testing parameters    | FPS_1  |  W  | FPS_W | FPS_W / FPS_1 |
| :-------------------: | :----: | :-: | :---: | :-----------: |
| 1280x720@yolov4-tiny  | 1,92   | 4   |  7,64 | 3,98          |
| 1920x1080@yolov4-tiny | 1,71   | 4   |  4,95 | 2,89          |
| 1280x720@yolov4-416   | 0,22   | 21  |  4,22 | 19,18         |
| 1920x1080@yolov4-416  | 0,22   | 21  |  4,14 | 18,82         |

Where:
* **FPS_1** - resulting FPS in a single node mode. Use script [src/video.py](src/video.py) to reproduce the results;
* **W** - number of worker instances used in a distributed mode;
* **FPS_W** - resulting FPS in a distributed mode.

To get more detailed results read [text version](report/out/diploma.pdf) of this work.
