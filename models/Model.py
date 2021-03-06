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
        self.classifier = torch.nn.Linear(768, n_classes)


    def forward(self, tokenized_seq):
        input_ids = self.roberta.encode(tokenized_seq)
        input_ids = input_ids[:512]
        # print(input_ids)
        # print(self.roberta.decode(input_ids))

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


def run_batch(model, batch_df, class_dict, criterion, device):
    seq_col, label_col = 0, 1
    target_classes, pred_classes=[], []
    for row in batch_df.itertuples(index=False):
        target_classes.append(class_dict[row[label_col]])
        
        pred_cls = model(row[seq_col])
        pred_classes.append(pred_cls[0])
        # break

    target_classes = torch.tensor(target_classes, dtype=torch.long).to(device)
    pred_classes = torch.stack(pred_classes)
    batch_loss = criterion(pred_classes, target_classes)
    print(batch_loss)
 
    return batch_loss, get_metrics(target_classes.cpu().numpy(), pred_classes.argmax(dim=1).cpu().numpy())


def train(model, train_loader, class_dict, criterion, optimizer, device):
    model.train()
    losses = []
    for batch_no, batch_df in enumerate(train_loader):
        model.zero_grad()
        batch_loss, _ = run_batch(model, batch_df, class_dict, criterion, device)  
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
        batch_loss, metrics = run_batch(model, batch_df, class_dict, criterion, device)
        losses.append(batch_loss)
        print("test batch_no:{}, batch_shape:{}, batch_loss:{}".format(batch_no, batch_df.shape, batch_loss))
        # break
    epoch_loss = torch.stack(losses).mean().item()
    return epoch_loss, metrics


def get_metrics(target_classes, pred_classes):
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    acc = accuracy_score(target_classes, pred_classes)
    precision = precision_score(target_classes, pred_classes, average="micro")
    recall = recall_score(target_classes, pred_classes, average="micro")
    return {"acc": acc, 
            "precision": precision, 
            "recall": recall, 
            "pred_classes": pred_classes, 
            "target_classes": target_classes}
    
