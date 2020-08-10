#!/bin/bash
#SBATCH --job-name=cp_adgan_newMasks
#SBATCH --output=cp_adgan_newMasks.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=190GB
#SBATCH --gres=gpu:Volta100:8
#SBATCH --nodelist=dgx-2
#SBATCH --cpus-per-task=16
#SBATCH --partition=deadline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

##################
#   CITYPERSON   #
##################
# train the detector on Cityperson
CACHEPATH="/home/vobecant/PhD/CSP/data/cache/cityperson/train_h50_adgan_newMasks"
SPECIF=cp_adgan_newMasks
LR=0.0006
python -u train_city_custom.py ${CACHEPATH} ${SPECIF} ${LR} >${SPECIF}.out