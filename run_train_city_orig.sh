#!/bin/bash
#SBATCH --job-name=csp_orig_higherLR
#SBATCH --output=csp_orig_higherLR.out
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
python -u train_city_orig.py >csp_orig_higherLR.out
