#!/bin/bash

#PBS -q qprod
#PBS -N traffic-sim
#PBS -l select=11:ncpus=128,walltime=02:30:00
#PBS -A OPEN-21-52

ml Rust SQLite Anaconda3

cd /home/sur096/ruth

source "/apps/all/Anaconda3/2021.05/etc/profile.d/conda.sh"
conda activate ox
python3 distributed-traffic-sim.py starting-cars_2000.pickle --departure-time="2021-10-25T18:01:00" --seed=660277 --n-samples=1000 --out=cars_2000.pickle

# exit
