#!/bin/bash

# Split tokenized data into sequence and family files
for x in "all" "train" "val" "test"; do
    for f in data/preprocess/tokenized/$x.txt; do
        cut -f1 -d',' "$f" > data/preprocess/tokenized/$(basename -s .txt "$f").sequence
        cut -f2 -d',' "$f" > data/preprocess/tokenized/$(basename -s .txt "$f").family
    done
done

# Generate class label dictionary file
awk '{print $0,0}' data/preprocess/tokenized/all.family | sort | uniq > data/preprocess/dictionaries/class_label_dict.txt