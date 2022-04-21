#!/bin/bash

# Binarize sequences
fairseq-preprocess \
	--only-source \
	--trainpref data/preprocess/tokenized/train.sequence \
    --validpref data/preprocess/tokenized/val.sequence \
    --testpref data/preprocess/tokenized/test.sequence \
	--destdir data/preprocess/binarized/input0 \
	--workers 60 \
	--srcdict data/preprocess/dictionaries/fragment_dict.txt \