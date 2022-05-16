import sys
sys.path.append("../scop_classification_by_PRoBERTa")

import torch
import numpy as np
import pandas as pd
import models.Model as Model
import models.dataloader as Dataloader
import utils as Utils

peak_lr=1e-05
batch_size=64
epochs=300
# warmup_updates=int(epochs*0.40) #40% epochs for warmup
device = "cuda" if torch.cuda.is_available() else "cpu"
out_filename = f"CW_{peak_lr}_{batch_size}_{epochs}_{device}"
print(out_filename)
# CW_1e-05_64_300_cuda

# data specific things
seq_col, label_col = 0, 1
df = pd.read_csv("data/splits/all.txt", header=None)
class_dict = {label:i for i, label in enumerate(df[label_col].unique())}
n_classes = len(class_dict)
print(f"n_classes: {n_classes}")

model = Model.Pooler(n_classes).to(device)

checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
model.load_state_dict(checkpoint['model_state_dict'])

@torch.no_grad()
def test(model, batch_df, class_dict, device):
    seq_col, label_col = 0, 1
    true_labels, pred_class_distributions=[], []
    for row in batch_df.itertuples(index=False):
        true_labels.append(class_dict[row[label_col]])
        
        y_pred = model(row[seq_col])
        y_pred_distribution = torch.nn.functional.softmax(y_pred)
        pred_class_distributions.append(y_pred_distribution.squeeze(0).cpu().numpy())
        # break

    return {"true_labels": true_labels,
            "pred_class_distributions": pred_class_distributions}


# evaluating validation set
val_loader = Dataloader.get_batched_data("data/preprocess/tokenized/val.txt", batch_size=0) #batch_size=0 means taking all
print(f"val data: {len(val_loader)}")
metrics = test(model, val_loader[0], class_dict, device)
print(f"Val: {metrics}")
Utils.save_as_pickle(metrics, f"outputs/predictions/{out_filename}_val_result.pkl")


# evaluating test set
# test_loader = Dataloader.get_batched_data("data/preprocess/tokenized/test.txt", batch_size=0) #batch_size=0 means taking all
# print(f"test data: {len(test_loader)}")
# metrics = test(model, criterion, test_loader, device)
# print(f"Test: {metrics}")
# Utils.save_as_pickle(metrics, f"outputs/predictions/{out_filename}_test_result.pkl")

