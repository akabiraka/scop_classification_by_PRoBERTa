#!/bin/bash

#SBATCH --job-name=pretrained
#SBATCH --output=/scratch/akabir4/scop_classification_by_PRoBERTa/outputs/argo_logs/pretrained-%j.output
#SBATCH --error=/scratch/akabir4/scop_classification_by_PRoBERTa/outputs/argo_logs/pretrained-%j.error
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --mem=32000MB

HEAD_NAME=protein_superfam_classification
PREFIX=superfam_pretrained
OUTPUT_DIR=outputs/models
DATA_DIR=data/preprocess/binarized/

ENCODER_EMBED_DIM=768
ENCODER_LAYERS=5
TOTAL_UPDATES=12500
WARMUP_UPDATES=3125
PEAK_LR=0.0025
MAX_SENTENCES=32
UPDATE_FREQ=64

NUM_CLASSES=2796
PATIENCE=5
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
    --save-dir "$CHECKPOINT_DIR" --save-interval-updates 100 --no-epoch-checkpoints \
    --log-format simple --log-interval 1000 2>&1 | tee -a "$LOG_FILE"

## could not install apex, so did not use lamb optimizer