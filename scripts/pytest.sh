#!/bin/bash
set -ex

PYTHON="${PYTHON:=$(which python)}"

echo "Using: ${PYTHON}"

mkdir -p logs

$PYTHON -m pytest -x --testmon --nbmake --overwrite "./notebooks"
$PYTHON -m mypy test
$PYTHON -m pytest -x --testmon --pylint
