import numpy as np 
import pandas as pd 
import glob
import torch

if __name__ == '__main__':
    sub = pd.read_csv('/home/ubuntu/repos/kaggle/melonama/data/sample_submission.csv')

    # 672x672 (without metadata) full images with color constancy & external data
    # Unique ID: 32-36 (0.934)
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

    # 256x256 (with metadata) full images with color constancy & external data 
    # Unique ID: 37-41 (0.927)
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

    # 224x224 (without metadata & external - cdeotte) cropped images without color constancy & without external data 
    # Unique ID: 55-59 (0.921)
    np_array_paths = [
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_0_224_0.9242182821734489.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_1_224_0.9117843862238233.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_2_224_0.8875828708802848.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_3_224_0.8916940744070858.npy',
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficient_net_fold_4_224_0.9017679248483987.npy'
    ]
    predictions = [np.load(path) for path in np_array_paths]
    predictions5 = sum(predictions) / len(predictions)
    predictions5 = torch.sigmoid(torch.tensor(predictions5)).numpy()


    # 512x512 (without metadata & external) with color constancy  
    # Unique ID: 65 (0.928)
    np_array_paths = [
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b6_fold_0_512_0.9082806113150564.npy',
    ]
    predictions = [np.load(path) for path in np_array_paths]
    predictions6 = sum(predictions) / len(predictions)
    predictions6 = torch.sigmoid(torch.tensor(predictions6)).numpy()


    # 384x384 (without metadata) with color constancy & external  
    # Unique ID: 67-71 (0.935)
    np_array_paths = [
        '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b3_fold_0_384_0.9392967794006613.npy',
        # '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b3_fold_1_384_0.9222934552052092.npy', 
        # '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b3_fold_2_384_0.9111322333305092.npy', 
        # '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b3_fold_3_384_0.8982232948704804.npy', 
        # '/home/ubuntu/repos/kaggle/melonama/data/output/efficientnet-b3_fold_4_384_0.9107568159730032.npy'
    ]
    predictions = [np.load(path) for path in np_array_paths]
    predictions7 = sum(predictions) / len(predictions)
    predictions7 = torch.sigmoid(torch.tensor(predictions7)).numpy()


    tabular_sub = pd.read_csv('/home/ubuntu/repos/kaggle/melonama/data/submission_tabular.csv').target.values
    predictions = 0.85*(
        #672x672           
        (0.25*predictions1) + (0.1*predictions2) + (0.1*predictions5) + (0.1*predictions3) + (0.25*predictions6) + (0.25*predictions7)
        )  + 0.15*(tabular_sub)

    sub['target'] = predictions

    sub.to_csv("/home/ubuntu/repos/kaggle/melonama/data/output/submission.csv", index=False)