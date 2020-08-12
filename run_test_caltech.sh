#!/bin/bash
STEP=${1}

for i in $(seq 1 ${STEP} 249); do
  START=${i}
  END=$(expr ${i} + ${STEP})
  EXPNAME="test_caltech_${START}-${END}"
  JOB="./jobs/${EXPNAME}.job"

  echo "#!/bin/bash
#SBATCH --job-name=tst_${START}-${END}
#SBATCH --output=${EXPNAME}.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=node-16,node-12
#SBATCH --cpus-per-task=16
#SBATCH --partition=deadline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

python -u test_caltech.py ${START} ${END} >${EXPNAME}.out" >${JOB}
  sbatch ${JOB}
  echo "Run job ${JOB}\n"
  sleep 2
done
