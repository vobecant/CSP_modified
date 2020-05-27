#!/bin/bash
#SBATCH --job-name=csp_paper_8gpu
#SBATCH --output=csp_paper_8gpu.out
#SBATCH --time=3-00:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

##################
#   CITYPERSON   #
##################
# train the detector on Cityperson
python -u train_city_orig_paper_8gpu.py >csp_paper_8gpu.out
