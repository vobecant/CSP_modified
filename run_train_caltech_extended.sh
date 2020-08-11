#!/bin/bash
#SBATCH --job-name=csp_caltech_extended
#SBATCH --output=csp_caltech_extended.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=190GB
#SBATCH --gres=gpu:Volta100:8
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

###############
#   CALTECH   #
###############
# train the detector on Caltech
CACHEPATH="data/cache/caltech/train_gt_ext"
SPECIF=caltech_extended
LR=0.0004
python -u train_city_custom.py ${CACHEPATH} ${SPECIF} ${LR} >csp_${SPECIF}.out