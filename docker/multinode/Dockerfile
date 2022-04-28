FROM ubuntu:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    build-essential gdal-bin libgdal-dev openmpi-bin libopenmpi-dev \
    python3 python3-dev python3-pip python3-setuptools python3-wheel vim

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV HOME="/root"
ENV PATH="${HOME}/.cargo/bin:${PATH}"

RUN python3 -m pip install -U pip setuptools wheel cython

RUN python3 -m pip install git+https://project_1147_bot:glpat-FVaHTJgWaoCFy4QwrYyk@code.it4i.cz/everest/ruth.git

# Install HyperQueue in multinode branch
ENV HQ_DIR="${HOME}/hyperqueue"
RUN git clone https://github.com/It4innovations/hyperqueue.git ${HQ_DIR}

WORKDIR ${HQ_DIR}
RUN git checkout -b multinode # TODO: use main when available
RUN cargo build --release

# add hyperqueu to PATH
ENV PATH="${HQ_DIR}/target/release:${PATH}"

ENTRYPOINT ["/bin/bash"]
