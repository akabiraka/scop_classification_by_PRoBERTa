import sys
sys.path.append("../scop_classification_by_PRoBERTa")

import torch
import pandas as pd
import models.Model as Model
import models.dataloader as Dataloader

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
criterion = torch.nn.CrossEntropyLoss()

checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
model.load_state_dict(checkpoint['model_state_dict'])


val_loader = Dataloader.get_batched_data("data/preprocess/tokenized/val.txt", batch_size=0) #batch_size=0 means taking all
val_loss, metrics = Model.test(model, val_loader, class_dict, criterion, device)
print(f"Val: {metrics}")

test_loader = Dataloader.get_batched_data("data/preprocess/tokenized/test.txt", batch_size=0) #batch_size=0 means taking all
test_loss, metrics = Model.test(model, test_loader, class_dict, criterion, device)
print(f"Test: {metrics}")


