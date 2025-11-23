# Makefile for conda environment: torch_env
# Usage:
#   make deps
#   make data
#   make train
#   make eval
#   make tb

SHELL := /usr/bin/env bash

CONDA_ENV = torch_env

.PHONY: deps data train eval tb

deps:
	@echo ">>> Activating conda env: $(CONDA_ENV)"
	conda activate $(CONDA_ENV) && pip install -r requirements.txt

data:
	conda activate $(CONDA_ENV) && python -m src.data_loader --ticker AAPL --start 2018-01-01 --end 2023-12-31

train:
	conda activate $(CONDA_ENV) && python -m src.train_dqn --config config.yaml

eval:
	conda activate $(CONDA_ENV) && python -m src.evaluate --checkpoint models/policy_latest.pth

tb:
	conda activate $(CONDA_ENV) && tensorboard --logdir logs/tensorboard
