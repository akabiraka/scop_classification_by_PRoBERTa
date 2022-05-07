import sys
sys.path.append("../scop_classification_by_PRoBERTa")

import torch
from fairseq.models.roberta import RobertaModel

class Pooler(torch.nn.Module):
    def __init__(self, n_classes, inner_dim=768, drop_prob=0.1) -> None:
        super(Pooler, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name_or_path="data/preprocess/pretrained_task_agnostic_model/", 
                                                          checkpoint_file="checkpoint_best.pt",
                                                          bpe="sentencepiece", 
                                                          sentencepiece_model="data/preprocess/pretrained_tokenization_model/m_reviewed.model")
        self.classifier = torch.nn.Sequential(torch.nn.Linear(768, inner_dim),
                                              torch.nn.Dropout(p=drop_prob),
                                              torch.nn.LeakyReLU(),
                                              torch.nn.Linear(inner_dim, n_classes))


    def forward(self, tokenized_seq):
        input_ids = self.roberta.encode(tokenized_seq)
        input_ids = input_ids[:512]
        # print(tokens)

        features = self.roberta.extract_features(input_ids) # Extract the last layer's features, [batch_size, tokens_len, 786]
        # print(features.shape)
        
        x = features[:, 0, :]  # take <CLS> token, [batch_size, 786]
        # print(x.shape)
 
        out = self.classifier(x)
        # print(out.shape) # [batch_size, n_classes]
        return out


# example usage
# seq = "▁M KVI FLKD VKG KG KKG EIKN VADG YAN NFL FKQ GLAI EAT P ANL KAL EAQ K"
# m = Pooler(2)
# print(m)
# out = m(seq)
# print(out)


def run_batch(model, batch_df, class_dict, criterion, device, print_meters=False):
    seq_col, label_col = 0, 1
    losses = []
    target_classes, pred_classes=[], []
    for row in batch_df.itertuples(index=False):
        target_cls = class_dict[row[label_col]]
        target_cls = torch.tensor([target_cls], dtype=torch.long).to(device)
        pred_cls = model(row[seq_col])
        
        per_item_loss = criterion(pred_cls, target_cls)
        losses.append(per_item_loss)
        
        pred_classes.append(pred_cls.argmax().item())
        target_classes.append(target_cls[0].item())
        # break
    batch_loss = torch.stack(losses).mean()
    if print_meters: 
        return batch_loss, get_metrics(target_classes, pred_classes)
    else: return batch_loss

def train(model, train_loader, class_dict, criterion, optimizer, device):
    model.train()
    losses = []
    for batch_no, batch_df in enumerate(train_loader):
        model.zero_grad()
        batch_loss=run_batch(model, batch_df, class_dict, criterion, device)  
        batch_loss.backward()
        optimizer.step() 
        losses.append(batch_loss)
        print("train batch_no:{}, batch_shape:{}, batch_loss:{}".format(batch_no, batch_df.shape, batch_loss))
        # break
    epoch_loss = torch.stack(losses).mean().item()
    return epoch_loss


@torch.no_grad()
def test(model, data_loader, class_dict, criterion, device):
    model.eval()
    losses = []
    for batch_no, batch_df in enumerate(data_loader):
        batch_loss, metrics = run_batch(model, batch_df, class_dict, criterion, device, print_meters=True)
        losses.append(batch_loss)
        print("test batch_no:{}, batch_shape:{}, batch_loss:{}".format(batch_no, batch_df.shape, batch_loss))
        # break
    epoch_loss = torch.stack(losses).mean().item()
    return epoch_loss, metrics


def get_metrics(target_classes, pred_classes):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, log_loss
    acc = accuracy_score(target_classes, pred_classes)
    precision = precision_score(target_classes, pred_classes)
    recall = recall_score(target_classes, pred_classes)
    loss = log_loss(target_classes, pred_classes)
    return {"acc": acc, "precision": precision, "recall": recall, "log_loss": loss}
    
