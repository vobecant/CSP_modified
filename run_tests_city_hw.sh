#!/bin/bash

MINEP=${1}
MAXEP=${2}
SPECIF=${3} # _lr0.0006_cp_cpl
MODELDIR=${4} # /home/vobecant/PhD/CSP/output/valmodels/city/hw/off_trnval_lr0.0006_cp_cpl
EXPNAME="test_city_hw_ep${MINEP}-${MAXEP}${SPECIF}"
JOB_FILE="./jobs/${EXPNAME}.job"
RESDIR="/home/vobecant/PhD/CSP/output/valresults/city/hw/off_trnval${SPECIF}"

echo "#!/bin/bash -l
#SBATCH --job-name=${EXPNAME}
#SBATCH --output=${EXPNAME}.out
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --exclude=node-10,node-12,node-16
#SBATCH --mem=20GB
#SBATCH --time=2-00:00:00

python -u test_city_hw.py ${MINEP} ${MAXEP} ${MODELDIR} > ${EXPNAME}.out
python -u /home/vobecant/PhD/CSP/eval_city/dt_txt2json.py ${RESDIR}
python /home/vobecant/PhD/CSP/eval_city/eval_script/eval_demo_allTrain.py ${RESDIR}" >${JOB_FILE}
echo "run job ${JOB_FILE}"
sbatch ${JOB_FILE}
echo ""