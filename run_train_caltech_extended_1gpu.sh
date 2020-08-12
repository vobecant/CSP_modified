#!/bin/bash
#SBATCH --job-name=csp_caltech_extended_run2
#SBATCH --output=csp_caltech_extended_run2.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=deadline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

###############
#   CALTECH   #
###############
# train the detector on Caltech
CACHEPATH="data/cache/caltech/train_gt_ext"
SPECIF=caltech_extended_run2
LR=1e-4
python -u train_caltech_extended.py ${CACHEPATH} ${SPECIF} ${LR} >csp_${SPECIF}.out