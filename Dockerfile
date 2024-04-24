FROM python:3.11-slim-buster

ENV CONFIG_FILENAME="config.json"

ENV RUSTUP_HOME="/usr/local/bin/rustup"
ENV CARGO_HOME="/usr/local/bin/cargo"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libboost-graph-dev \
    libpthread-stubs0-dev \
    gdal-bin libgdal-dev \
    git curl \
    python3 python3-dev python3-pip python3-setuptools python3-wheel vim

WORKDIR /usr/local/bin

# install RUST
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="${CARGO_HOME}/bin:${PATH}"
ENV PATH="${RUSTUP_HOME}/bin:${PATH}"

RUN pip install --upgrade pip
RUN python3 -m pip install -U pip setuptools wheel
RUN python3 -m pip install cython ipython

RUN python3 -m pip install git+https://github.com/It4innovations/ruth.git

WORKDIR /app
RUN mkdir -p inputs
RUN mkdir -p outputs

# ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ruth-simulator-conf --config-file=${CONFIG_FILENAME} run