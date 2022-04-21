
#### Workflow
* Download pretrained tokenizer: [URL](https://drive.google.com/drive/folders/1lJkG4IAWxSs8mGqSk-MjsaBQFV4Y3dhq)
* Download and generate fasta: `python generators/DownloadCleanFasta.py`
* Tokenize all, train, val, test sequences: `python generators/Tokenizer.py`
* Preprocess the tokenized sequence and class label: `bash generators/preprocess_seq_and_label.sh`
* Binarize sequences: `bash generators/binarize_seqs.sh`
* Binarize class labels: `bash generators/binarize_labels.sh`