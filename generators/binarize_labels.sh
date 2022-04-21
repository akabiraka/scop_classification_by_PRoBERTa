#!/bin/bash

# Binarize labels
fairseq-preprocess \
	--only-source \
	--trainpref data/preprocess/tokenized/train.family \
	--validpref data/preprocess/tokenized/val.family \
	--testpref data/preprocess/tokenized/test.family \
	--destdir data/preprocess/binarized/label \
	--workers 60 \
	--srcdict data/preprocess/dictionaries/class_label_dict.txt