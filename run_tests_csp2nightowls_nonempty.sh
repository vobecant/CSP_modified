#!/bin/bash

WDIR=${1}
SPECIF=${2}
EXPNAME="test_csp2nightowls_nonempty_${SPECIF}"
JOB_FILE="./jobs/${EXPNAME}.job"
RESDIR="/home/vobecant/PhD/CSP/output/valresults/nightowls/h/${SPECIF}"

echo "#!/bin/bash -l
#SBATCH --job-name=${EXPNAME}
#SBATCH --output=${EXPNAME}.out
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --exclude=node-10,node-12,node-01
#SBATCH --mem=20GB
#SBATCH --time=2-00:00:00

python -u test_csp2nightowls_nonempty.py ${WDIR} > ${EXPNAME}.out
python -u /home/vobecant/PhD/CSP/eval_city/dt_txt2json.py ${RESDIR}
python -u /home/vobecant/PhD/CSP/eval_nightowls/eval.py ${RESDIR}" >${JOB_FILE}
echo "run job ${JOB_FILE}"
sbatch ${JOB_FILE}
echo ""