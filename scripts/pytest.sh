#!/bin/bash
set -ex

PYTHON="${PYTHON:=$(which python)}"

echo "Using: ${PYTHON}"

mkdir -p logs

$PYTHON -m pytest -x --testmon --nbmake --overwrite "./examples"
$PYTHON -m pytest -x --testmon --pylint --cov-fail-under 98
$PYTHON -m mypy tests
