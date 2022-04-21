#!/bin/bash

#SBATCH --job-name=proberta_scop
#SBATCH --output=/scratch/akabir4/scop_classification_by_PRoBERTa/outputs/argo_logs/proberta_scop-%j.output
#SBATCH --error=/scratch/akabir4/scop_classification_by_PRoBERTa/outputs/argo_logs/proberta_scop-%j.error
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --mem=16000MB

HEAD_NAME=protein_superfam_classification
PREFIX=superfam
OUTPUT_DIR=outputs/models
DATA_DIR=data/preprocess/binarized/

ENCODER_EMBED_DIM=768
ENCODER_LAYERS=5
TOTAL_UPDATES=1000
WARMUP_UPDATES=40
PEAK_LR=0.0025
MAX_SENTENCES=1
UPDATE_FREQ=2

NUM_CLASSES=2
PATIENCE=3
ROBERTA_PATH=data/preprocess/pretrained_task_agnostic_model/checkpoint_best.pt

TOKENS_PER_SAMPLE=512
MAX_POSITIONS=512

PREFIX="$PREFIX.DIM_$ENCODER_EMBED_DIM.LAYERS_$ENCODER_LAYERS"
PREFIX="$PREFIX.UPDATES_$TOTAL_UPDATES.WARMUP_$WARMUP_UPDATES"
PREFIX="$PREFIX.LR_$PEAK_LR.BATCH_$MAX_SENTENCES.PATIENCE_$PATIENCE"

CHECKPOINT_DIR="$OUTPUT_DIR/$PREFIX/checkpoints"
LOG_FILE="$OUTPUT_DIR/$PREFIX/$PREFIX.log"

mkdir -p "$CHECKPOINT_DIR"

## this runs
CUDA_VISIBLE_DEVICES=0 fairseq-train  $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --max-positions $MAX_POSITIONS \
    --batch-size $MAX_SENTENCES \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --encoder-embed-dim $ENCODER_EMBED_DIM --encoder-layers $ENCODER_LAYERS \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --bpe sentencepiece \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --patience $PATIENCE \
    --update-freq $UPDATE_FREQ \
    --save-dir "$CHECKPOINT_DIR" --save-interval 1 --save-interval-updates 100 --keep-interval-updates 5 \
    --log-format simple --log-interval 1000 2>&1 | tee -a "$LOG_FILE"

## could not install apex, so did not use lamb optimizer