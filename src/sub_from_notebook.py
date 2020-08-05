import numpy as np 
import pandas as pd 
import glob
import torch
from scipy.stats import rankdata
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required paramaters
    parser.add_argument("--path")
    args = parser.parse_args()

    sub = pd.read_csv('/home/ubuntu/repos/kaggle/melonama/data/sample_submission.csv')

    predictions = torch.sigmoid(torch.tensor(np.load(args.path))).numpy()
    
    sub['target'] = predictions

    sub_path = "/home/ubuntu/repos/kaggle/melonama/data/output/submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"file saved at {sub_path}.")