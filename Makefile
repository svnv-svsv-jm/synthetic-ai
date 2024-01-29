help:
	@cat Makefile

.EXPORT_ALL_VARIABLES:

# create an .env file to override the default settings
-include .env
export $(shell sed 's/=.*//' .env)


.PHONY: build docs

# ----------------
# default settings
# ----------------
# user
LOCAL_USER:=$(shell whoami)
LOCAL_USER_ID:=$(shell id -u)
# project
PROJECT_NAME?=sai
# python
PYTHON?=python
PYTHON_EXEC?=python -m
PYTHONVERSION?=3.10.10
PYTEST?=pytest
SYSTEM=$(shell python -c "import sys; print(sys.platform)")
# poetry
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
POETRY=poetry
# docker
DOCKER?=docker
DOCKERFILE?=Dockerfile
REGISTRY=registry.com
MEMORY=8g
SHM=4g
CPUS=1
DOCKER_COMMON_FLAGS=--cpus=$(CPUS) --memory=$(MEMORY) --shm-size=$(SHM) --network=host --volume $(PWD):/workdir -e LOCAL_USER_ID -e LOCAL_USER
IMAGE=$(REGISTRY)/$(PROJECT_NAME)
IMAGE_PYTHON=/$(PROJECT_NAME)/bin/python
CUDA_DEVICE=1
BUILD_CMD=$(DOCKER) build -t $(REGISTRY)/$(PROJECT_NAME) --build-arg project_name=$(PROJECT_NAME) -f $(DOCKERFILE) .



# -----------
# install project's dependencies
# -----------
install-init:
	$(PYTHON_EXEC) pip install --upgrade pip
	$(PYTHON_EXEC) pip install --upgrade poetry setuptools virtualenv setuptools_rust

install: install-init
	$(PYTHON_EXEC) $(POETRY) install --no-cache


# -----------
# testing
# -----------
pytest:
	bash scripts/pytest.sh

test: pytest


# ---
# git
# ---
# squash all commits before rebasing, see https://stackoverflow.com/questions/25356810/git-how-to-squash-all-commits-on-branch
git-squash: BRANCH=main
git-squash:
	git reset $(shell git merge-base $(BRANCH) $(shell git branch --show-current))
	git add -A
	git commit -m "squashed commit"

# Locally delete branches that have been merged
git-clean:
	bash scripts/git-clean.sh


# -----------
# docker
# -----------
# build project's image
build: DOCKER_HOST= 
build:
	$(BUILD_CMD)

build-nohup: DOCKER_HOST= 
build-nohup:
	mkdir -p logs || echo "Folder already exists."
	nohup $(BUILD_CMD) 2>&1 > logs/build-$(shell echo "$(DOCKERFILE)" | tr "/" "-").log &

# launch bash session within a container
bash: DOCKER_HOST= 
bash:
	$(DOCKER) run --rm -it $(DOCKER_COMMON_FLAGS) \
		--name $(PROJECT_NAME)-bash \
		-t $(REGISTRY)/$(PROJECT_NAME) bash

# run
run: CONFIG=main
run: RANDOM=$(shell bash -c 'echo $$RANDOM')
run: CONTAINER_NAME=run-$(CONFIG)-$(RANDOM)
run: SCRIPT=experiments/main.py
run: DOCKER_FLAGS=
run: GPU_FLAGS=
run: OVERRIDE=
run: NOW=$(shell date '+%Y-%m-%d_%H:%M:%S')
run: CMD=$(IMAGE_PYTHON) -u /workdir/$(SCRIPT) --config-name $(CONFIG) $(OVERRIDE)
run:
	$(DOCKER) run --rm -it $(DOCKER_FLAGS) $(DOCKER_COMMON_FLAGS) \
		$(GPU_FLAGS) \
		--name $(PROJECT_NAME)-$(CONTAINER_NAME) \
		-t $(REGISTRY)/$(PROJECT_NAME) \
		$(CMD)

# clean dangling images
clean: DOCKER_HOST= 
clean:
	$(DOCKER) system prune

# WARNING: cleans everything, even images you may want to keep
clean-all: DOCKER_HOST= 
clean-all:
	$(DOCKER) rmi $(docker images -f dangling=true -q)