#!/bin/bash
#
#SBATCH --job-name=mono
#SBATCH --output=res_taller_1_paralelismo.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00
#SBATCH --men-per-cpu=100

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./mono
