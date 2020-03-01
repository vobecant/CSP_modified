#!/bin/bash

EXP_NAMES=( eccv_1P eccv_2P)


for EXP_NAME in "${EXP_NAMES[@]}"
do
    EXPNAME="t_${EXP_NAME}_valMR"
    echo "Run experiment ${EXP_NAME}"


    job_file="${EXPNAME}.job"
    echo "run job_file ${job_file}"

    echo "#!/bin/bash
#SBATCH --job-name=$EXPNAME
#SBATCH --output=${EXPNAME}.out
#SBATCH --time=3-00:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --partition=deadline
#SBATCH --exclude=node-17
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

##################
#   CITYPERSON   #
##################
# train the detector on Cityperson
python -u train_city_valMR_eccv.py ${EXP_NAME}

# test the detector on images from Cityperson test split and convert the detections to JSON file
python -u test_city_eccv.py ${EXP_NAME}
RES_FOLDER=/home/vobecant/PhD/CSP/output/valresults/city_valMR_eccv/h/off_${EXP_NAME}
python /home/vobecant/PhD/CSP/eval_city/dt_txt2json.py ${RES_FOLDER}

# evaluate the detections
# TODO: get the best performing one and copy it to 'best_val.hdf5' to the right directory
python /home/vobecant/PhD/CSP/eval_city/eval_script/eval_demo_eccv.py ${RES_FOLDER}" > ${job_file}
    sbatch ${job_file}

	echo ""

done
