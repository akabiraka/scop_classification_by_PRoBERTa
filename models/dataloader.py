import sys
sys.path.append("../scop_classification_by_PRoBERTa")

import numpy as np
import pandas as pd

def get_batched_data(data_path, batch_size=0):
    df =  pd.read_csv(data_path, header=None)
    df = df.sample(frac=1).reset_index(drop=True)
    # print(f"data shape: {data.shape}")

    if batch_size==0: split_num=1 #taking all
    else: split_num=np.ceil(len(df) / batch_size)
    batched_data=np.array_split(df, split_num)
    print(f"Total batches: {len(batched_data)}" )
    return batched_data