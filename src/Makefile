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


.PHONY: clean
clean:
	find . -name *.out.* | xargs --no-run-if-empty rm
