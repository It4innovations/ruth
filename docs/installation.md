# Installation instruction

It is important to note that the simulator is primarly meant to be running on linux machine and for large city simulations computer cluster.

Up to the version v1.1 it was also possible to run it on mac smoothly. Nevertheless, this had always been only for developing purposes. Dependencies introduced in v1.1 make it impossible. We provide workaround based on docker which helps to overcome this issue.

## Installation

``` sh
# requirements
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    git curl \
    build-essential gdal-bin libgdal-dev openmpi-bin libopenmpi-dev \
    python3 python3-dev python3-virtualenv python3-pip python3-setuptools python3-wheel

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# create and activate python virtual environment
virtualenv venv
source venv/bin/activate

# within the venv
# install and update python dependencies
python3 -m pip install -U pip setuptools wheel cython

# install ruth - the traffic simulator
python3 -m pip install git+https://github.com/It4innovations/ruth.git
```

## Test run

* activate virtual environment with `ruth` installed
* use the files in `benchmarks/od-matrix/`


``` sh
ruth-simulator --departure-time="2021-06-16 07:00:00" --k-alternatives=4 --nproc=8 --out=simulation_record.pickle --seed=7 rank-by-prob-delay benchmarks/od-matrices/INPUT-od-matrix-10-vehicles-town-resolution.parquet benchmarks/od-matrices/prob-profile-for-2021-06-20T23:59:00+02:00--2021-06-27T23:59:00+02:00.mem 70 500
```

