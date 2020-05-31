#!/bin/bash
#SBATCH --job-name=csp_1P_run2
#SBATCH --output=csp_hard_1P_run2.err
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
SPECIF=eccv_1P_hard_allTrain
LR=0.0008
ORDINAL=2
python -u train_city_orig.py ${SPECIF} ${LR} ${ORDINAL} >csp_hard_1P_run2.out
