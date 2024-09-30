#!/bin/bash
#SBATCH --job-name="RN101_100K_400e"
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH --account=research-ceg-wm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=T.Jia@tudelft.nl

# Load modules:
module load miniconda3
module load cuda/11.7


# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

conda activate vissl_env
srun python tools/object_detection_benchmark_1.py \
    --config-file /scratch/tjian/PythonProject/deep_plastic_Flux_SSL/configs/config/benchmark/object_detection/COCOInstance/R50_C4_COCO/tiles_224/Flux/FRC_101.yaml \
     --num-gpus 1 SOLVER.MAX_ITER 100250 TEST.EVAL_PERIOD 1002 SOLVER.IMS_PER_BATCH 4 MODEL.BACKBONE.FREEZE_AT 0 MODEL.WEIGHTS /scratch/tjian/PythonProject/deep_plastic_Flux_SSL/checkpoint/train_weights/RN101_100K_400e/detectron2/RN101_100K_400e.torch OUTPUT_DIR /scratch/tjian/PythonProject/deep_plastic_Flux_SSL/checkpoint/train_weights/RN101_100K_400e/SSL_F0/



/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate