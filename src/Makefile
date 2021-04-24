PROJECT = bs-diploma
REGISTRY ?= master:5000

.PHONY: build
build: build/worker build/client

.PHONY: push
push: push/worker push/client


WORKER_CONFIG ?= yolov3.cfg
WORKER_WEIGHTS ?= yolov3.weights

CLIENT_CLASSES ?= coco.names
CLIENT_VIDEO ?= video_720_15s.mp4
CLIENT_TIMEOUT_SEC ?= 10

.PHONY: deploy
deploy:
	WORKER_IMAGE=$(WORKER_IMAGE) \
	WORKER_CONFIG=$(WORKER_CONFIG) \
	WORKER_WEIGHTS=$(WORKER_WEIGHTS) \
	CLIENT_IMAGE=$(CLIENT_IMAGE) \
	CLIENT_CLASSES=$(CLIENT_CLASSES) \
	CLIENT_VIDEO=$(CLIENT_VIDEO) \
	CLIENT_TIMEOUT_SEC=$(CLIENT_TIMEOUT_SEC) \
		docker stack deploy \
			--prune \
			--compose-file deploy/docker-compose.yml $(PROJECT)

.PHONY: stop
stop:
	docker stack rm $(PROJECT)


WORKER_PATH = worker
WORKER_IMAGE = $(REGISTRY)/$(PROJECT)-worker

ARCH = $(shell uname --machine)
ifeq (aarch64, $(ARCH))
	WORKER_REQUIREMENTS=requirements-arm64.txt
else
	WORKER_REQUIREMENTS=requirements.txt
endif

.PHONY: build/worker
build/worker:
	docker build \
		--build-arg REQUIREMENTS=$(WORKER_REQUIREMENTS) \
		--tag $(WORKER_IMAGE) \
		$(WORKER_PATH)

.PHONY: push/worker
push/worker:
	docker push $(WORKER_IMAGE)


CLIENT_PATH = client
CLIENT_IMAGE = $(REGISTRY)/$(PROJECT)-client

.PHONY: build/client
build/client:
	docker build --tag $(CLIENT_IMAGE) $(CLIENT_PATH)

.PHONY: push/client
push/client:
	docker push $(CLIENT_IMAGE)


.PHONY: clean
clean:
	find . -name *.out.* | xargs --no-run-if-empty rm
	find . -regextype posix-extended -regex '.+\.(cfg|weights)' | xargs --no-run-if-empty rm
