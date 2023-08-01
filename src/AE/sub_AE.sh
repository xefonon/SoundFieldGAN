#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J real_AE
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 20:00
# request 24GB of system-memory
#BSUB -R "rusage[mem=24GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
#BSUB -o ../../log/log-%J-%I.out
#BSUB -e ../../log/log-%J-%I.err
# -- end of LSF options --

source /work3/xenoka/miniconda3/bin/activate tf_2.7
module load cuda/11.3
module load cudnn/v8.2.0.53-prod-cuda-11.3
module load tensorrt/v8.0.1.6-cuda-11.3

python run_SF_AE.py --use_wandb --epochs 1000 --config_file './config.yaml' --resume_training
