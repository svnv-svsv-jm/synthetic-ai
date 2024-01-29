# Developer guide

This document is intended for developers who wish to contribute to the project.

## Tools

Format your code on save by installing the [Black extension](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) (VSCode users).

## Technologies

This projct uses `make` commands to facilitate users and developers. So `make` sure you have that installed!

This project contains a `Dockerfile`. You can build and develop in a container running this image if you experience problems in simply installing this project in a virtual environment. You can launch a development container by running `make dev-container`. Check [here](https://code.visualstudio.com/docs/remote/create-dev-container) why you'd like to do this.

We use [pytest](https://docs.pytest.org/en/7.1.x/) with the following plugins: [pylint](https://pylint.pycqa.org/en/latest/) and [mypy](http://www.mypy-lang.org/). [Coverage](https://coverage.readthedocs.io/en/6.4.4/) is also enforced.

## Dependencies

In order to install this project's dependencies, just run:

```bash
make install
```

Which uses [Poetry](https://python-poetry.org/) behind the scenes. [Poetry](https://python-poetry.org/) can make it a lot easier to handle dependencies. It relies on a `pyproject.toml` file with a `tool.poetry.dependencies` section.

This is the only command we want to support to install the project. For the user, it will always be `make install`, we can change the backend of the command if we need to.

With `poetry`, you can also specify a different source for each dependency. For example:

```TOML
torch = [
    { version = "*", source = "pytorch", platform = "linux" },
    { version = "^2.0.0", source = "pypi", platform = "darwin" },
]

[[tool.poetry.source]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
priority = "explicit"
```

Here (above), `poetry` installs `torch` from the source `"pytorch"` if we are on Linux, or from PyPi if we are on Mac. This is just an example of how we can leverage `poetry` to have a flexible, fully customized dependecy list.

### Add new dependencies

To add new dependencies, run:

```bash
poetry add <package-name>
```

which will edit the `pyproject.toml` automatically.

Alternatively, manually edit the `pyproject.toml` file, then run `poetry lock`.

### Project installation

The packages developed in this project will automatically be installed by `poetry` in editable mode, and once installed it can be imported normally as `import <package>` from anywhere on your file system. If you find yourself writing `import ..<package>.<module>` etc., especially in testing, something is wrong.

## Project's layout

We use the "src" layout from the official [PyTest doc](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html).

```bash
src/
 |--package_1/
    |--__init__.py
    |--module/
        |--__init__.py
        |--some_file.py
 |--package_2/
    |--__init__.py
 |-- ...
```

Develop any new package within the `src` folder. For example, if creating a package named `cool`, create the folder `src/cool/`. Make sure it contains the `__init__.py` file: `src/cool/__init__.py`.

```bash
src/
 |--cool
    |--__init__.py
 |--package_2
    |--__init__.py
 |-- ...
```

The folder `test` will be used to test the code residing in `src`. The `test` folder will contain the `conftest.py` file and then should mimic the layout of the `src` folder as much as possible for all tests. This way, it will be easy to find a specific test for a certain function residing in specific (sub)module.

Eventually, a folder `test/integration` (or any other name) can be used to design cross-module tests.

Notebooks for quick development or to showcase our code can be stored in `notebooks/` or in `examples/`. They may import all modules in `src` if needed.

We can use the `results` folder in case we want our repository to store useful results, figures, etc. although it is not recommended to store huge amount of data here on Gitlab.

Code that can be re-used to run specific experiments can be placed in `experiments/`.

### Package all files, not only .py

With `poetry`, a `MANIFEST.in` is not necessary, and you can specify what to include and/or exclude directly in the `pyproject.toml` file.

For example:

```TOML
[tool.poetry]
name = "myproject"
version = "0.1.0"
description = "My project description"
authors = ["Your Name"]
license = "LICENSE"
readme = "README.md"
include = ["*.py", "data/*.json"]  # Specify which files to include
exclude = ["test/*"]  # Specify which files to exclude
```

### Pretrained models

For development, we can push and exchange model checkpoints in the `checkpoints/` folder. However, these will not end up in the final package that is going to be created out of this repository. To include them, these checkpoints have to be in the `src/<package-name>` folder, and be included in the `include` keyword of the `pyproject.toml` file.

## Testing

This project uses `pytest` for test-driven development (TDD). Make sure you're familiar with it: <https://docs.pytest.org/en/7.1.x/>

To run all your tests, run:

```bash
python -m pytest
# test notebooks too
pytest --nbmake --overwrite "./examples"
```

You can also just run:

```bash
make pytest
```

Also checkout [this script](./scripts/pytest.sh).

### conftest.py

The file `conftest.py` can be used to create PyTest fixtures that can be shared by all tests.

### Run single test locally

In a `test_*.py` file, place the following lines at the top and bottom of the file:

```python
# TOP (can also be placed under `if __name__ == "__main__"`)
import pytest
from loguru import logger
import sys

# code for tests
def test_1() -> None:
    ...
def test_2() -> None:
    ...

# BOTTOM
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
```

This will make the file runnable. When run, this file will run `pytest.main()` on itself, thus the tests declared here (and only those) will be run.

## Useful git commands (not important)

There are some useful commands in the `Makefile`, just to avoid having to type them from scratch each time.

To clear all merged branches (local branches that do not have a remote counterpart anymore), **from the `main` branch**, run:

```bash
make git-clean
```

## CI/CD Pipeline

We use a `.gitlab-ci.yml` file to automatically test our code when pushing. We can test the source code here and the notebooks.

FYI, you can also run the CI/CD locally. For example:

```bash
gitlab-runner exec docker <step-name>
```

## Project's image

The project's image can be built locally as follows: `make build`.
