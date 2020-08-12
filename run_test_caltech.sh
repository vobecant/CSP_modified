#!/bin/bash
START=${1}
END=${2}
EXPNAME="test_caltech_${START}-${END}"
JOB="${EXPNAME}.job"

echo "#!/bin/bash
#SBATCH --job-name=csp_tst_caltech
#SBATCH --output=csp_tst_caltech.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=190GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=node-16,node-12
#SBATCH --cpus-per-task=16
#SBATCH --partition=deadline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

python -u test_caltech.py ${START} ${END} >${EXPNAME}.out">${JOB}
sbatch ${JOB}