#!/bin/bash

SPECIF=${1}
CACHE=${2}
LR=${3}
WEIGHTS=${4}
EXPNAME="train_nightowls_${SPECIF}"
JOB_FILE="./jobs/${EXPNAME}.job"

echo "#!/bin/bash
#SBATCH --job-name=no_hw_${SPECIF}
#SBATCH --output=csp_hw_nightowls_${SPECIF}.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:Volta100:8
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

# train the detector on Nightowls
python -u train_nightowls_custom_hw.py ${SPECIF} ${CACHE} ${LR} ${WEIGHTS} > csp_hw_nightowls_${SPECIF}.out" >${JOB_FILE}
echo "run job ${JOB_FILE}"
sbatch ${JOB_FILE}
echo ""
