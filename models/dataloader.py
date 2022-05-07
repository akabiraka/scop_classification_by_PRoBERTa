import sys
sys.path.append("../scop_classification_by_PRoBERTa")

import numpy as np
import pandas as pd

def get_batched_data(data_path, batch_size=0):
    data =  pd.read_csv(data_path, header=None)
    # print(f"data shape: {data.shape}")

    if batch_size==0: split_num=1 #taking all
    else: split_num=np.ceil(len(data) / batch_size)
    batched_data=np.array_split(data, split_num)
    print(f"Total batches: {len(batched_data)}" )
    return batched_data