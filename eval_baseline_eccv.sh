#!/bin/bash

#conda deactivate
#ml purge
#conda activate csp
#ml cuDNN/7.0.5.15-fosscuda-2018a
#ml CUDA/10.0.130
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/software/CUDA/10.0.130/lib64:/home/vobecant/miniconda3/envs/csp/lib/

FROM=$1
TO=$2

EXPNAME="e_base_csp_${FROM}-${TO}"
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
#SBATCH --reservation=vobecant_1
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

##################
#   CITYPERSON   #
##################
# test the detector (with best val loss) on images from Cityperson test split and convert them to JSON file
python -u eval_city_eccv_baseline.py ${FROM} ${TO}
RES_FOLDER=/home/vobecant/PhD/CSP/output/valresults/city_valMR_eccv/h/off_baseline
python /home/vobecant/PhD/CSP/eval_city/dt_txt2json.py ${RES_FOLDER}

RES_FILE=/home/vobecant/PhD/CSP/output/valresults/city_val/h/off_${EXP_NAME}
python /home/vobecant/PhD/CSP/eval_city/dt_txt2json.py ${RES_FILE}

# evaluate the detections
python /home/vobecant/PhD/CSP/eval_city/eval_script/eval_demo_eccv.py ${RES_FOLDER}" > ${job_file}
    sbatch ${job_file}

	echo ""
