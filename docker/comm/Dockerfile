FROM ubuntu:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    build-essential gdal-bin libgdal-dev openmpi-bin libopenmpi-dev \
    python3 python3-dev python3-pip python3-setuptools python3-wheel vim

ENV INSTALL_DIR=/install
ENV WORKDIR=/workdir


# install RUST
ENV RUSTUP_HOME="${INSTALL_DIR}/rustup"
ENV CARGO_HOME="${INSTALL_DIR}/cargo"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="${CARGO_HOME}/bin:${PATH}"

ENV VIRTUAL_ENV="${INSTALL_DIR}/venv"
RUN python3 -m pip install virtualenv
RUN virtualenv ${VIRTUAL_ENV}

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# install the simulator
RUN python3 -m pip install -U pip setuptools wheel cython

RUN python3 -m pip install git+https://project_1147_bot:glpat-FVaHTJgWaoCFy4QwrYyk@code.it4i.cz/everest/ruth.git


# install EvKit
ENV EVKIT_DIR="${INSTALL_DIR}/evkit"

RUN git clone https://project_1281_bot:glpat-yhU_xHPPqi72ryGWxzwF@code.it4i.cz/everest/evkit.git ${EVKIT_DIR}
WORKDIR ${EVKIT_DIR}
RUN git checkout -b comm origin/comm

RUN python3 -m pip install .


# install HyperQueue in comm branch
ENV HQ_DIR="${INSTALL_DIR}/hyperqueue"
RUN git clone https://github.com/It4innovations/hyperqueue.git ${HQ_DIR}

WORKDIR ${HQ_DIR}
RUN git checkout v0.9.0
RUN cargo build --release

# install the HyperQueue Python API
RUN python3 -m pip install maturin

WORKDIR ${HQ_DIR}/crates/pyhq
RUN maturin develop

# add hyperqueu to PATH
ENV PATH="${HQ_DIR}/target/release:${PATH}"

WORKDIR ${WORKDIR}

ENTRYPOINT ["/bin/bash"]
