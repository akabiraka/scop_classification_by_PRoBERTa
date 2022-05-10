#!/usr/bin/sh

#SBATCH --job-name=pbertA
#SBATCH --output=/scratch/akabir4/scop_classification_by_PRoBERTa/outputs/argo_logs/pbertA-%j.out
#SBATCH --error=/scratch/akabir4/scop_classification_by_PRoBERTa/outputs/argo_logs/pbertA-%j.err
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

##cpu jobs
#SBATCH --partition=all-LoPri
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000MB

##GPU jobs
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --mem=16000MB
#SBATCH --time=1-24:00

##python generators/Tokenizer.py
python models/train_val.py

