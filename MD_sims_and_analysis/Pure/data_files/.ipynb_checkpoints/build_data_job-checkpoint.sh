#!/bin/bash

#SBATCH --no-requeue
#SBATCH --job-name=db_983
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=def-moosavi5
#SBATCH --time=0-0:30 # D-HH:MM
#SBATCH --mem=800
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load Dependencies
module load scipy-stack/2023b rdkit/2024.09.6 ambertools/25.0
source ~/mixture-env/bin/activate

python ../../build_data.py -l "C11.lt" -s "CCCC(C)O:1.0" -rho 0.803409445 -b 40 -o . -name mixid_983 --seed 42
