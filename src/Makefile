PROJECT = bs-diploma
REGISTRY ?= master:5000


.PHONY: deploy
deploy:
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


.PHONY: clean
clean:
	find . -name *.out.* | xargs --no-run-if-empty rm
