PROJECT = bs-diploma
REGISTRY ?= master:5000

DASK_WORKER_IMAGE ?= $(DASK_IMAGE)
DASK_SCHEDULER_IMAGE ?= $(DASK_IMAGE)

.PHONY: deploy
deploy:
	DASK_WORKER_IMAGE=$(DASK_WORKER_IMAGE) \
	DASK_SCHEDULER_IMAGE=$(DASK_SCHEDULER_IMAGE) \
		docker stack deploy \
			--prune \
			--compose-file deploy/docker-compose.yml $(PROJECT)

.PHONY: stop
stop:
	docker stack rm $(PROJECT)


DASK_PATH = deploy/dask
DASK_IMAGE = $(REGISTRY)/$(PROJECT)-dask

.PHONY: build/dask
build/dask:
	docker build --tag $(DASK_IMAGE) $(DASK_PATH)

.PHONY: push/dask
push/dask:
	docker push $(DASK_IMAGE)


.PHONY: clean
clean:
	find . -name *.out.* | xargs --no-run-if-empty rm
