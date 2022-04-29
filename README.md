# SCOP classification using PRoBERTa from sequence

#### Downloads
* Download pretrained tokenizer and vocab: [URL](https://drive.google.com/drive/folders/1lJkG4IAWxSs8mGqSk-MjsaBQFV4Y3dhq)
* Download pretrained task agnostic model: [URL](https://drive.google.com/drive/u/2/folders/1TbFjyRfbkLgJ_rlvO1SFB-ZvwQyykvK7)

#### Workflow
* Download and generate fasta: `python generators/DownloadCleanFasta.py`
* Tokenize all, train, val, test sequences: `python generators/Tokenizer.py`
* Preprocess the tokenized sequence and class label: `bash generators/preprocess_seq_and_label.sh`
* Binarize sequences: `bash generators/binarize_seqs.sh`
* Binarize class labels: `bash generators/binarize_labels.sh`

#### Train-test
* Finetune the model: `bash models/PRoBERTa_finetune_superfamily.sh`
* Evaluate the finetuned model: `python models/eval_superfam_classification.py`

#### Todo
* Create two datasets: all and all_exclude_less_than_10_data
* Finetune and evaluate on both of this set.
* Fucus on classes having <10 data points when evaluating model finetuned on all.