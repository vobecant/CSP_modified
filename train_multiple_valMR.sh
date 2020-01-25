#!/bin/bash

EXP_NAMES=( baseline 1P halfP blurred)


for EXP_NAME in "${EXP_NAMES[@]}"
do
    EXPNAME="csp_${EXP_NAME}_valMR"
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
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

# train the detector on Cityperson
python -u train_city_valMR.py ${EXP_NAME}

# test the detector on images from Cityperson test split
python -u test_city_bestValMR.py ${EXP_NAME}
python /home/vobecant/PhD/CSP/eval_city/dt_txt2json.py /home/vobecant/PhD/CSP/output/valresults/city_valMR/h/off_${EXP_NAME}

# evaluate the detections
python /home/vobecant/PhD/CSP/eval_city/eval_script/eval_demo_valMR.py ${EXP_NAME}


# finetune the detector for Precarious Pedestrians
python -u train_precarious_valMR.py ${EXP_NAME}

# test the detector on images Precarious Pedestrians test split
python -u test_precarious_finetuned_valMR.py "${EXP_NAME}"


cd /home/vobecant/PhD/CSP/eval_precarious
python dt_txt2json.py /home/vobecant/PhD/CSP/output/valresults/precarious_valMR/h/off_${EXP_NAME}_finetuned

cd eval_script
python eval_precarious_finetuned_valMR.py ${EXP_NAME}" > ${job_file}
    sbatch ${job_file}

	echo ""

done
