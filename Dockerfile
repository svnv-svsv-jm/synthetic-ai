FROM python:3.10.10

ARG project_name=sai

ENV PROJECT_NAME=$project_name

# Create workdir and copy dependency files
RUN mkdir -p /workdir
COPY pyproject.toml /workdir/pyproject.toml
COPY Makefile /workdir/Makefile
COPY scripts /workdir/scripts
COPY README.md /workdir/README.md
COPY LICENSE /workdir/LICENSE
COPY src /workdir/src
COPY poetry.lock /workdir/

# Change shell to be able to easily activate virtualenv
SHELL ["/bin/bash", "-c"]
WORKDIR /workdir

# Install project
RUN bash scripts/install-system-packages.sh && \
    bash scripts/setup-virtualenv.sh

# TensorBoard
EXPOSE 6006
# Jupyter Notebook
EXPOSE 8888

# Handle user-permissions using GOSU (https://denibertovic.com/posts/handling-permissions-with-docker-volumes/): The entrypoint script `entrypoint.sh` is needed to log you in within the container at runtime: this means that any file you create in the container will belong to your user ID, not to root's, thus solving all those annoying permission-related issues

# Set entrypoint and default container command
ENTRYPOINT ["/workdir/scripts/entrypoint.sh"]
