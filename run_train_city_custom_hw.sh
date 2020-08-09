#!/bin/bash
#SBATCH --job-name=cp_cpl
#SBATCH --output=cp_cpl.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=190GB
#SBATCH --gres=gpu:Volta100:8
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

##################
#   CITYPERSON   #
##################
# train the detector on Cityperson
CACHEPATH="/home/vobecant/PhD/CSP/data/cache/cityperson/train_h50_cpl"
SPECIF=cp_cpl
LR=0.0006
python -u train_city_custom_hw.py ${CACHEPATH} ${SPECIF} ${LR} >csp_${SPECIF}.out