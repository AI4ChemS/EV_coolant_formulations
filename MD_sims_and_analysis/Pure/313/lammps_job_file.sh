#!/bin/bash

#SBATCH --no-requeue
#SBATCH --job-name=983_npt
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=def-moosavi5
#SBATCH --time=0-12:00 # D-HH:MM
#SBATCH --mem=800
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load StdEnv/2023 intel/2023.2.1 openmpi/4.1.5 lammps-omp/20240829

srun lmp -sf omp -in lmp_npt.in -var mytemp 313.15 -var data mixid_983_system.data -var settings mixid_983_system.in.settings -var name mixid_983
