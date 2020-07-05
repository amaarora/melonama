import numpy as np 
import pandas as pd 
import glob
import torch

if __name__ == '__main__':
    sub = pd.read_csv('/home/ubuntu/repos/kaggle/melonama/data/sample_submission.csv')

    # 672x672 (without metadata) full images with color constancy & external data
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

    #256x256 (with metadata) full images with color constancy & external data 
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
 
    # 384x384 (with metadata) full images with color constancy & external data: 0.927
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


    # 352x352 (without metadata) cropped images with color constancy & external data: 0.924
    np_array_paths = [
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_0_352_0.9193366138144706.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_1_352_0.9144833933700429.npy',
    ]
    predictions = [np.load(path) for path in np_array_paths]
    predictions4 = sum(predictions) / len(predictions)
    predictions4 = torch.sigmoid(torch.tensor(predictions4)).numpy()

    # 224x224 (without metadata & external - cdeotte) cropped images without color constancy & without external data : 0.918
    np_array_paths = [
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_0_224_0.9242182821734489.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_1_224_0.9160147261849977.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_2_224_0.8768109802592561.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_3_224_0.8920647690257042.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_4_224_0.8995808691146435.npy'
    ]
    predictions = [np.load(path) for path in np_array_paths]
    predictions5 = sum(predictions) / len(predictions)
    predictions5 = torch.sigmoid(torch.tensor(predictions5)).numpy()


    # 512x512 (without metadata & external - cdeotte) cropped images without color constancy & without external data : 0.918
    np_array_paths = [
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b2_fold_0_480_0.939480330130959.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b2_fold_1_480_0.9207740239406298.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b2_fold_2_480_0.9216102368042023.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b2_fold_3_480_0.9205924804202257.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b2_fold_4_480_0.9208070143598686.npy'
    ]
    predictions = [np.load(path) for path in np_array_paths]
    predictions6 = sum(predictions) / len(predictions)
    predictions6 = torch.sigmoid(torch.tensor(predictions6)).numpy()

    predictions = ((0.4*predictions1) + (0.15*predictions2) + (0.1*predictions3) + (0.1*predictions4) + (0.1*predictions5) + (0.15*predictions6))  
    sub['target'] = predictions
    sub.to_csv("/home/ubuntu/repos/kaggle/melonama/data/output/submission.csv", index=False)