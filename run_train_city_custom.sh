#!/bin/bash
#SBATCH --job-name=csp_baseline_reasonable
#SBATCH --output=csp_baseline_reasonable.err
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
CACHEPATH="/home/vobecant/PhD/CSP/data/cache/cityperson/train_h50_reasonable"
SPECIF=baseline_reasonable
LR=0.0008
python -u train_city_custom.py ${CACHEPATH} ${SPECIF} ${LR} >csp_${SPECIF}.out