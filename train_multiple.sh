#!/bin/bash

EXP_NAMES=( baseline 1P halfP blurred)


for EXP_NAME in "${EXP_NAMES[@]}"
do
    EXPNAME="cityperson_${EXP_NAME}"
    echo "Test experiment ${EXP_NAME}"


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

python -u train_city_val.py ${EXP_NAME}
python -u train_precarious.py ${EXP_NAME}
python -u test_precarious_finetuned.py "${EXP_NAME}"

cd /home/vobecant/PhD/CSP/eval_precarious
python dt_txt2json.py /home/vobecant/PhD/CSP/output/valresults/precarious/h/off_${EXP_NAME}_finetuned

cd eval_script
python eval_precarious.py ${EXP_NAME}" > ${job_file}
    sbatch ${job_file}

	echo ""

done
