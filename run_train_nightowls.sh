#!/bin/bash
#SBATCH --job-name=no_all
#SBATCH --output=csp_nightowls_all.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:Volta100:8
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

##################
#   CITYPERSON   #
##################
# train the detector on Cityperson
SPECIF='all'
python -u train_nightowls_orig.py ${SPECIF} >csp_nightowls_all.out
