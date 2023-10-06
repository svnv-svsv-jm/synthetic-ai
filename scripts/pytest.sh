#!/bin/bash
set -ex

PYTHON="${PYTHON:=$(which python)}"

echo "Using ${PYTHON}"

mkdir -p logs

$PYTHON -m pytest -x --testmon --mypy --pylint
$PYTHON -m pytest -x --testmon --nbmake --overwrite "./examples"
