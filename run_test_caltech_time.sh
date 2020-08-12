#!/bin/bash

START=${1}
END=${2}
TIME=${3}
EXPNAME="test_caltech_run2_${START}-${END}"
JOB="./jobs/${EXPNAME}.job"
echo "Start testing epochs ${START}-${END} at ${TIME}"

echo "#!/bin/bash
#SBATCH --job-name=tst_${START}-${END}
#SBATCH --output=${EXPNAME}.err
#SBATCH --time=3-00:00:00
#SBATCH --begin=${TIME}
#SBATCH --mem=20GB
#SBATCH --gres=Volta100:1
#SBATCH --exclude=node-16,node-12
#SBATCH --cpus-per-task=16
#SBATCH --partition=deadline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

python -u test_caltech.py ${START} ${END} >${EXPNAME}.out" >${JOB}
sbatch ${JOB}
echo "Run job ${JOB}\n"

