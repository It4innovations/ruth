#!/usr/bin/bash
#SBATCH --job-name Ruth_Bench
#SBATCH --account DD-23-154
#SBATCH --partition qcpu
#SBATCH --nodes 2
#SBATCH --time 48:00:00
#SBATCH -o slurm.%N.%j.out      # STDOUT
#SBATCH -e slurm.%N.%j.err      # STDERR

ml purge
ml Python/3.10.8-GCCcore-12.2.0
ml GCC/12.2.0
ml SQLite/3.39.4-GCCcore-12.2.0
ml HDF5/1.14.0-gompi-2022b
ml CMake/3.24.3-GCCcore-12.2.0
ml Boost/1.81.0-GCC-12.2.0

#ml Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate
#rm -rf workers
python3 ruth/zeromq/bench.py workers_1_nodes_250k_full
