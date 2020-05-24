#!/bin/bash

MINEP=${1}
MAXEP=${2}
EXPNAME="test_allTrain_ep${MINEP}-${MAXEP}"
JOB_FILE="./jobs/${EXPNAME}.job"
RESDIR="/home/vobecant/PhD/CSP/output/valresults/city/h/off_trnval"

echo "#!/bin/bash -l
#SBATCH --job-name=${EXPNAME}
#SBATCH --output=${EXPNAME}.out
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --exclude=node-12
#SBATCH --mem=20GB
#SBATCH --time=2-00:00:00

python -u test_city_eccv_allTrain.py ${MINEP} ${MAXEP} > ${EXPNAME}.out
python -u /home/vobecant/PhD/CSP/eval_city/dt_txt2json.py ${RESDIR}
python /home/vobecant/PhD/CSP/eval_city/eval_script/eval_demo_allTrain.py ${RESDIR}" >${JOB_FILE}
echo "run job ${JOB_FILE}"
sbatch ${JOB_FILE}
echo ""
