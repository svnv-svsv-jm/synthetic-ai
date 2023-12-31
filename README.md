# Synthetic A.I.

This repository offers different synthetic A.I. applications and implementations.

## Installation via virtual environment

To run the notebooks, you may want to follow these steps.

### Pre-requisites

The project contains a `Makefile` so that useful but long commands can be run without too much thinking. Besides, `mypy`, `pylint` are also configured.

A `.env` file lets you choose the project's name, the Python version and more stuff. Check the file please.

Make sure you have Python `>=3.9` (`3.10.10` recommended).

Create a virtual environment with any tool you prefer.

#### Create a virtual environment

Use any tool to create a virtual environment with the indicated Python version.

With [Pyenv](https://github.com/pyenv/pyenv) (**recommended**, here is the [installer](https://github.com/pyenv/pyenv-installer)), it is very easy to install the Python version you want, create a virtual environment and activate it.

Once Pyenv is installed, run:

```bash
pyenv install 3.10.10
pyenv virtualenv 3.10.10 <project-name>
pyenv activate <project-name>
```

#### Getting Started

Once the virtual environment is active, install all dependencies by running:

```bash
make install
```

For this command to work, you have to make sure that the `python` command resolves to the correct virtual environment (the one you just created).

## Installation via Docker

Build project's image as follows:

```bash
make build
```

The present folder will be mounted at `/workdir`.

## Testing

Run:

```bash
make pytest
```

## Usage

See the [notebooks](./notebooks) folder.
