#!/bin/bash

EXP_NAMES=( baseline 1P halfP) #  blurred


for EXP_NAME in "${EXP_NAMES[@]}"
do
    EXPNAME="tPP_${EXP_NAME}"
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
#SBATCH --exclude=node-12
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

##################
#   PRECARIOUS   #
##################
# finetune the detector for Precarious Pedestrians
python -u train_precarious_valMR.py ${EXP_NAME}

# test the detector on images Precarious Pedestrians test split and convert the detections to JSON file
python -u test_precarious_finetuned_valMR.py "${EXP_NAME}"
cd /home/vobecant/PhD/CSP/eval_precarious
python dt_txt2json.py /home/vobecant/PhD/CSP/output/valresults/precarious_valMR/h/off_${EXP_NAME}_finetuned

# evaluate the detections
cd eval_script
python eval_precarious_finetuned_valMR.py ${EXP_NAME}" > ${job_file}
    sbatch ${job_file}

	echo ""

done
