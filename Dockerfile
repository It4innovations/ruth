FROM python:3.11-slim-buster

ENV VEHICLES_FILENAME="INPUT-od-matrix-10-vehicles-town-resolution.parquet"
ENV PROB_PROFILE_FILENAME="prob-profile-for-2021-06-20T23-59-00+02-00--2021-06-27T23-59-00+02-00.mem"

ENV DEPARTURE_TIME="2021-06-16T07:00:00"
ENV K_ALTERNATIVES="4"
ENV NPROC="8"
ENV SEED="7"
ENV WALLTIME_S="600"
ENV NEAR_DISTANCE="70"
ENV N_SAMPLES="500"


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
ENTRYPOINT ruth-simulator --walltime-s=${WALLTIME_S} --departure-time=${DEPARTURE_TIME} --k-alternatives=${K_ALTERNATIVES} --nproc=${NPROC} --out=/app/outputs/simulation_record.pickle --seed=${SEED} rank-by-prob-delay /app/inputs/${VEHICLES_FILENAME} --prob_profile_path /app/inputs/${PROB_PROFILE_FILENAME} ${NEAR_DISTANCE} ${N_SAMPLES}