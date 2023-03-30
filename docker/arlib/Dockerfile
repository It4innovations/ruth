
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


# prepare ruth environment
ENV RUTH_HOME=/ruth
WORKDIR ${RUTH_HOME}

# install py arlib
RUN git clone https://project_1485_bot:glpat-gp8mAP8LjPbPGdUuZBdQ@code.it4i.cz/everest/py-arlib.git arlib

WORKDIR ${RUTH_HOME}/arlib

RUN git submodule init
RUN git submodule update

# NOTE: a dirty fix of external library arlib. It counts at SIGSTKSZ which is not available in new glibc.
#       Now it can be compiled and used. plus it should not influence the library as it is among tests.
# TODO: remove once fixed
RUN sed -i 7355s/SIGSTKSZ/4196464/ external/arlib/external/catch/catch.hpp


RUN mkdir build

WORKDIR ${RUTH_HOME}/arlib/build

RUN cmake ../
RUN make

# set the environment variables in order to work
ENV LD_LIBRARY_PATH=${RUTH_HOME}/arlib/build/
ENV PYTHONPATH=${RUTH_HOME}

# install ruth depencies
ENV RUTH_HOST_HOME=.

RUN pip install -U pip setuptools wheel

ENV TMP=/tmp
COPY ${RUTH_HOST_HOME}/requirements.txt ${TMP}

RUN pip install -r ${TMP}/requirements.txt

COPY ${RUTH_HOST_HOME} ${RUTH_HOME}
WORKDIR ${RUTH_HOME}

RUN pip install .

ENTRYPOINT ["/bin/bash"]
