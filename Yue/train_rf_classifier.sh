#!/bin/bash -l
#SBATCH --output=./outfile/train_rf_classifier.%A_%a.out
#SBATCH --error=./errfile/train_rf_classifier.%A_%a.err
#SBATCH --job-name=train_rf_classifier
#SBATCH --array=1-3
#SBATCH --time=100:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu

hostname
date
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

PARAMFILE=/cellar/users/y8qin/Data/image_hackathon/morpho-type/Yue/train_rf_classifier.param
PARAM=$(awk "NR==$SLURM_ARRAY_TASK_ID" $PARAMFILE)
echo $PARAM

python -u /cellar/users/y8qin/Data/image_hackathon/morpho-type/Yue/train_rf_classifier.py $PARAM

date
