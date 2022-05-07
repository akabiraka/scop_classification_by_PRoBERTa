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
* Finetune the model: `sbatch models/PRoBERTa_finetune_superfamily.sh`
* Evaluate the finetuned model: `python models/eval_superfam_classification.py`

#### Analyze
* To vizualize the training progress: `tensorboard --logdir=outputs/tensorboard_runs/`
* To download the runs outputs: `scp -r akabir4@argo.orc.gmu.edu:/scratch/akabir4/scop_classification_by_PRoBERTa/outputs/tensorboard_runs/* outputs/tensorboard_runs/`