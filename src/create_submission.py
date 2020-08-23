import numpy as np 
import pandas as pd 
import glob
import torch
from scipy.stats import rankdata

if __name__ == '__main__':
    sub = pd.read_csv('/home/ubuntu/repos/kaggle/melonama/data/sample_submission.csv')
  
    np_array_paths = [
        # files output from training script
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b6_fold_0_512_0.9226915336106267.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b6_fold_1_512_0.900193720344129.npy', 
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b6_fold_2_512_0.8808248627551896.npy', 
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b6_fold_3_512_0.9153925564433958.npy', 
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b6_fold_4_512_0.907285738644841.npy'
    ]
    predictions = [np.load(path) for path in np_array_paths]    
    predictions = sum(predictions) / len(predictions)
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    sub['target'] = predictions
    sub.to_csv("/home/ubuntu/repos/kaggle/melonama/data/output/submission.csv", index=False)