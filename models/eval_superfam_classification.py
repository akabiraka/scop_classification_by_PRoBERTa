import sys
sys.path.append("../scop_classification_by_PRoBERTa")
import pandas as pd
import numpy as np
import torch
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens

seq_col=0
label_col=1
pred_col=2
batch_size=2
has_cuda=torch.cuda.device_count()>0
classification_head="protein_superfam_classification"
inp_file_path = "data/preprocess/tokenized/test.txt"
out_file_path = "outputs/eval_result/predictions_on_test.txt"

data = pd.read_csv(inp_file_path, header=None)

model = RobertaModel.from_pretrained(model_name_or_path="outputs/models/superfam.DIM_768.LAYERS_5.UPDATES_1000.WARMUP_40.LR_0.0025.BATCH_1.PATIENCE_3/checkpoints/", 
                                    checkpoint_file="checkpoint_best.pt", 
                                    data_name_or_path="data/preprocess/binarized/",
                                    bpe="sentencepiece", 
                                    sentencepiece_model="data/preprocess/pretrained_tokenization_model/m_reviewed.model")                                    
if has_cuda: model.cuda()
model.eval()      
# print(model)                              

split_num = int(len(data) / batch_size)
batched_data=np.array_split(data, split_num)
print("Total batches: " + str(len(batched_data)))

preds_df = pd.DataFrame(columns=[seq_col, label_col, pred_col])
for count, batch_df in enumerate(batched_data):
    batch=collate_tokens([torch.cat((model.encode(tokens), torch.ones(512, dtype = torch.long)))[:512]
            for tokens in batch_df[seq_col]], pad_idx=1)

    logprobs = model.predict(classification_head, batch)
    preds = model.task.label_dictionary.string(logprobs.argmax(dim=1) + model.task.label_dictionary.nspecial)
    batch_df[pred_col] = preds.split()
    preds_df = preds_df.append(batch_df, ignore_index=True)

    if (has_cuda):
        torch.cuda.empty_cache()
    
    print(f"Batch {count+1} completed.")

preds_df = preds_df.rename(columns={seq_col : "Tokenized_Sequence", label_col : "Family", pred_col : "Predicted"})
preds_df.to_csv(out_file_path, sep=",", index = False)

n_correct = np.where(preds_df["Family"]==preds_df["Predicted"], 1, 0).sum()
print("Accuracy: " + str(n_correct / len(preds_df)))