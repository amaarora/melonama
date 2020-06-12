import os
import pandas as pd 
from sklearn import model_selection
from tqdm import tqdm

if __name__ == '__main__':
    input_path = "/home/ubuntu/repos/kaggle/melonama/data/"
    kf = model_selection.StratifiedKFold(n_splits=8)
    df = pd.read_csv(os.path.join(input_path, 'train.csv'))
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    targets = df.target.values
    for fold, (train_index, test_index) in enumerate(kf.split(X=df, y=targets)):
        df.loc[test_index, 'kfold'] = fold
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)
