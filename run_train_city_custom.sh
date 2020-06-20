#!/bin/bash
#SBATCH --job-name=csp_occluded_fixedRatio
#SBATCH --output=csp_1P_occluded_fixedRatio.err
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
CACHEPATH="/home/vobecant/PhD/CSP/data/cache/cityperson_trainValTest/train_h50_1P_occluded_fixedRatio_dgx"
SPECIF=1P_occluded_fixedRatio
LR=0.0008
python -u train_city_custom.py ${CACHEPATH} ${SPECIF} ${LR} >csp_${SPECIF}.out