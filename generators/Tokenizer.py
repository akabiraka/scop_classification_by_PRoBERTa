import sys
sys.path.append("../scop_classification_by_PRoBERTa")
import pandas as pd
import os
from Bio import SeqIO
import sentencepiece as spm
from generators.IGenerator import IGenerator

class Tokenizer(IGenerator):
    def __init__(self) -> None:
        super(Tokenizer, self).__init__()
        self.fastas_dir = "data/fastas/"
        self.model = spm.SentencePieceProcessor()
        self.model.load("data/preprocess/pretrained_tokenization_model/m_reviewed.model")
        self.task = "SF" #superfamily classification

    def do(self, pdb_id, chain_id, region):
        fasta_file = self.fastas_dir+pdb_id+chain_id+region+".fasta"
        seq_record = next(SeqIO.parse(fasta_file, "fasta"))
        seq_tokened = self.model.encode_as_pieces(str(seq_record.seq))
        seq_tokened = " ".join(seq_tokened)
        return seq_tokened

    def do_distributed(self, i, df, out_file_path=None):
        row = df.loc[i]
        pdb_id, chain_and_region = row["FA-PDBID"].lower(), row["FA-PDBREG"]
        if chain_and_region.find(",")!=-1: return
        chain_and_region = chain_and_region.split(":")
        chain_id, region = chain_and_region[0], chain_and_region[1]
        if len(chain_id)>1: return

        seq_tokened = self.do(pdb_id, chain_id, region)
        label = str(row[self.task])

        with open(out_file_path, "a") as f:
            f.write(f"{seq_tokened},{label}\n")


    def do_linear(self, df, n_rows_to_skip, n_rows_to_evalutate, out_file_path=None):
        if out_file_path!=None and os.path.exists(out_file_path):
            os.remove(out_file_path)
        for i, row in df.iterrows():
            if i+1 <= n_rows_to_skip: continue
            self.do_distributed(i, df, out_file_path)
            if i+1 >= n_rows_to_skip+n_rows_to_evalutate: break
        

t = Tokenizer()
t.do_linear(pd.read_csv("data/splits/all.txt"), 0, 40000, "data/preprocess/tokenized/all_cleaned.txt")
# t.do_linear(pd.read_csv("data/splits/train_24538.txt"), 0, 40000, "data/preprocess/tokenized/train.txt")
# t.do_linear(pd.read_csv("data/splits/val_4458.txt"), 0, 40000, "data/preprocess/tokenized/val.txt")
# t.do_linear(pd.read_csv("data/splits/test_5862.txt"), 0, 40000, "data/preprocess/tokenized/test.txt")
