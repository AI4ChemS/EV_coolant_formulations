#!/bin/bash

#SBATCH --no-requeue
#SBATCH --job-name=db_977
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

python ../../build_data.py -l "C11.lt" "C26.lt" -s "CCCC(C)O:0.2541899106651536" "C(CCCl)CCl:0.7458100893348465" -rho 1.042971664 -b 40 -o . -name mixid_977 --seed 42
