#!/bin/bash

#conda deactivate
#ml purge
#conda activate csp
#ml cuDNN/7.0.5.15-fosscuda-2018a
#ml CUDA/10.0.130
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/software/CUDA/10.0.130/lib64:/home/vobecant/miniconda3/envs/csp/lib/


EXP_NAMES=( baseline 1P halfP blurred)


for EXP_NAME in "${EXP_NAMES[@]}"
do
    EXPNAME="csp_${EXP_NAME}"
    echo "Test experiment ${EXP_NAME}"


    job_file="${EXPNAME}.job"
    echo "run job_file ${job_file}"

    echo "#!/bin/bash
#SBATCH --job-name=$EXPNAME
#SBATCH --output=${EXPNAME}.out
#SBATCH --time=3-00:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

##################
#   CITYPERSON   #
##################
# test the detector (with best val loss) on images from Cityperson test split and convert them to JSON file
python -u test_city_bestVal.py ${EXP_NAME}
RES_FILE=/home/vobecant/PhD/CSP/output/valresults/city_val/h/off_${EXP_NAME}
python /home/vobecant/PhD/CSP/eval_city/dt_txt2json.py ${RES_FILE}

# evaluate the detections
python /home/vobecant/PhD/CSP/eval_city/eval_script/eval_demo_val.py ${RES_FILE}

##################
#   PRECARIOUS   #
##################
# test the detector on images Precarious Pedestrians test split
python -u test_precarious_finetuned.py "${EXP_NAME}"

cd /home/vobecant/PhD/CSP/eval_precarious
python dt_txt2json.py /home/vobecant/PhD/CSP/output/valresults/precarious/h/off_${EXP_NAME}_finetuned

cd eval_script
python eval_precarious_finetuned.py ${EXP_NAME}" > ${job_file}
    sbatch ${job_file}

	echo ""

done
