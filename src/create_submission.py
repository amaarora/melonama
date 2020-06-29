import numpy as np 
import pandas as pd 
import glob
import torch

if __name__ == '__main__':
    sub = pd.read_csv('/home/ubuntu/repos/kaggle/melonama/data/sample_submission.csv')

    # 672x672 (without metadata)
    np_array_paths = [
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_0_672_0.9312387721627744.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_1_672_0.9136423504768555.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_2_672_0.9131815004659831.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_4_672_0.9180649751506806.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_3_672_0.8942021784224183.npy',
    ]
    predictions = [np.load(path) for path in np_array_paths]
    predictions1 = sum(predictions) / len(predictions)
    predictions1 = torch.sigmoid(torch.tensor(predictions1)).numpy()

    #256x256 (with metadata)
    np_array_paths = [
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_0_256_0.9247324845739281.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_1_256_0.9038870463525716.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_2_256_0.9040074557315937.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_3,4_256_0.9001970059825113.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_3,4_256_0.8815525531920487.npy'
        ]
    predictions = [np.load(path) for path in np_array_paths]
    predictions2 = sum(predictions) / len(predictions)
    predictions2 = torch.sigmoid(torch.tensor(predictions2)).numpy()

    # 384x384 (with metadata)
    np_array_paths = [
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_0_384_0.9306738004113618.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_1_384_0.916072911542388.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_2_384_0.9167372701855461.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_3_384_0.8967300002366136.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_4_384_0.9124704260879057.npy',
    ]
    predictions = [np.load(path) for path in np_array_paths]
    predictions3 = sum(predictions) / len(predictions)
    predictions3 = torch.sigmoid(torch.tensor(predictions3)).numpy()


    predictions = ((0.6*predictions1) + (0.15*predictions2) + (0.25*predictions3))
    sub['target'] = predictions
    sub.to_csv("/home/ubuntu/repos/kaggle/melonama/data/output/submission.csv", index=False)