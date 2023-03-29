FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libboost-graph-dev \
    libpthread-stubs0-dev \
    gdal-bin libgdal-dev \
    git curl \
    python3 python3-dev python3-pip python3-setuptools python3-wheel vim

# install RUST
ENV RUSTUP_HOME="${INSTALL_DIR}/rustup"
ENV CARGO_HOME="${INSTALL_DIR}/cargo"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="${CARGO_HOME}/bin:${PATH}"

# install simulator
ENV WORKDIR=/workdir

RUN python3 -m pip install -U pip setuptools wheel
RUN python3 -m pip install cython ipython

WORKDIR ${WORKDIR}
ENTRYPOINT ["/bin/bash"]
