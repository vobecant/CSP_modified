#!/bin/bash

EXP_NAMES=( blurred) #( baseline 1P halfP) #  blurred


for EXP_NAME in "${EXP_NAMES[@]}"
do
    EXPNAME="e_${EXP_NAME}_valMR"
    echo "Run experiment ${EXP_NAME}"


    job_file="${EXPNAME}.job"
    echo "run job_file ${job_file}"

    echo "#!/bin/bash
#SBATCH --job-name=$EXPNAME
#SBATCH --output=${EXPNAME}.out
#SBATCH --time=3-00:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=node-12,node-10
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

##################
#   CITYPERSON   #
##################

# test the detector on images from Cityperson test split and convert the detections to JSON file
python -u test_city_valSet.py ${EXP_NAME}
RES_FILE=/home/vobecant/PhD/CSP/output/valresults/city_valMR/h/off_${EXP_NAME}
python /home/vobecant/PhD/CSP/eval_city/dt_txt2json.py ${RES_FILE}

# evaluate the detections
# TODO: get the best performing one and copy it to 'best_val.hdf5' to the right directory
python /home/vobecant/PhD/CSP/eval_city/eval_script/eval_demo_valMR.py ${RES_FILE}" > ${job_file}
    sbatch ${job_file}

	echo ""

done
