import sys
sys.path.append("../scop_classification_by_PRoBERTa")
import os
import torch
import pandas as pd
import models.dataloader as Dataloader
import models.Model as Model
from torch.utils.tensorboard import SummaryWriter

init_lr=0.0025
batch_size=16
epochs=100
device = "cuda" if torch.cuda.is_available() else "cpu"
out_filename = f"Model_{init_lr}_{batch_size}_{epochs}_{device}"
print(out_filename)


# data specific things
seq_col, label_col = 0, 1
df = pd.read_csv("data/preprocess/tokenized/all.txt", header=None)
class_dict = {label:i for i, label in enumerate(df[label_col].unique())}
n_classes = len(class_dict)
print(f"n_classes: {n_classes}")

model = Model.Pooler(n_classes).to(device)
# print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.01)
writer = SummaryWriter(f"outputs/tensorboard_runs/{out_filename}")

# load the AUC/loss based model checkpoint 
if os.path.exists(f"outputs/models/{out_filename}.pth"):
    checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    prev_n_epochs = checkpoint['epoch']
    print(f"Previously trained for {prev_n_epochs} number of epochs...")
    
    # train for more epochs with new lr
    start_epoch = prev_n_epochs+1
    n_epochs = 10
    new_lr=0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=new_lr, weight_decay=.01)
    print(f"Train for {n_epochs} more epochs...")


best_loss = torch.inf
for epoch in range(1, epochs+1):
    train_loader = Dataloader.get_batched_data("data/preprocess/tokenized/train.txt", batch_size)
    val_loader = Dataloader.get_batched_data("data/preprocess/tokenized/val.txt")
    train_loss = Model.train(model, train_loader, class_dict, criterion, optimizer, device)
    val_loss, metrics = Model.test(model, val_loader, class_dict, criterion, device)

    crnt_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch: {epoch:03d}, crnt_lr: {crnt_lr:.5f}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, acc: {metrics['acc']:.3f}")
    writer.add_scalar('train loss',train_loss,epoch)
    writer.add_scalar('val loss',val_loss,epoch)
    writer.add_scalar('acc',metrics["acc"],epoch)
    writer.add_scalar('precision',metrics["precision"],epoch)
    writer.add_scalar('recall',metrics["recall"],epoch)
    

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, f"outputs/models/{out_filename}.pth")