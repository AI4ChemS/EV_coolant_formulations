#!/bin/bash

#SBATCH --no-requeue
#SBATCH --job-name=983_trj
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=def-moosavi5
#SBATCH --time=0-6:00 # D-HH:MM
#SBATCH --mem=800
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load StdEnv/2023 intel/2023.2.1 openmpi/4.1.5 lammps-omp/20240829 rdkit

L=$(python ../../../ensemble_averager_cli.py ../mixid_983_313.15_npt_thermostyle.txt -p "Volume" -T 313 --tolerance 0.005 --tol-units "percent" --plot --save-plot mixid_983_Volume.png --json mixid_983_Volume.json --print-only "Lx") # get equillibirum box lenght
echo "Equillibrium box size $L"

srun lmp -sf omp -in lmp_trj.in -var mytemp 313.15 -var data mixid_983_313.15_annealed.data -var settings mixid_983_system.in.settings -var name mixid_983 -var BoxLength $L
