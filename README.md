# ruth

<p align="center">
    <img width="60%" src="docs/images/ruth.svg?sanitize=true">
</p>

A python library for routing on OSM map based on [osmnx](https://github.com/gboeing/osmnx) and [probduration](https://code.it4i.cz/everest/py-probduration).

## Documentation

The detailed documentation is available at: https://it4innovations.github.io/ruth/

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
python3 -m venv venv
source venv/bin/activate

# within the venv
# install and update python dependencies
python3 -m pip install -U pip setuptools wheel cython

# install ruth - the traffic simulator
python3 -m pip install git+https://github.com/It4innovations/ruth.git
```

## Test Run

* activate virtual environment with `ruth` installed
* use the files in `benchmarks/od-matrix/`

``` sh
ruth-simulator --departure-time="2021-06-16 07:00:00" --k-alternatives=4 --nproc=8 --out=simulation_record.pickle --seed=7 rank-by-prob-delay INPUT-od-matrix-10-vehicles-town-resolution.parquet --prob_profile_path=prob-profile-for-2021-06-20T23:59:00+02:00--2021-06-27T23:59:00+02:00.mem 70 500
```

