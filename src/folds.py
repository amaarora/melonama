import os
import pandas as pd 
from sklearn import model_selection
from tqdm import tqdm
import random
import numpy as np
from collections import Counter, defaultdict


def stratified_group_k_fold(X, y, groups, k, seed=None):
    """
    Usage:    
    
    train_x = pd.read_csv('../data/train.csv')
    train_y = train_x.target.values
    groups = np.array(train_x.patient_id.values)
        for fold_ind, (dev_ind, val_ind) in enumerate(stratified_group_k_fold(train_x, train_y, groups, k=5)):
    #     from pdb import set_trace; set_trace()
        dev_y, val_y = train_y[dev_ind], train_y[val_ind]
        train_x.loc[val_ind, 'kfold'] = fold_ind
        dev_groups, val_groups = groups[dev_ind], groups[val_ind]
        
        assert len(set(dev_groups) & set(val_groups)) == 0
        
        distrs.append(get_distribution(dev_y))
        index.append(f'development set - fold {fold_ind}')
        distrs.append(get_distribution(val_y))
        index.append(f'validation set - fold {fold_ind}')

    display('Distribution per class:')
    pd.DataFrame(distrs, index=index, columns=[f'Label {l}' for l in range(np.max(train_y) + 1)])
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def get_distribution(y_vals):
        y_distr = Counter(y_vals)
        y_vals_sum = sum(y_distr.values())
        return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]


if __name__ == '__main__':
    input_path = "/home/ubuntu/repos/kaggle/melonama/data/"
    kf = model_selection.StratifiedKFold(n_splits=5)
    df = pd.read_csv(os.path.join(input_path, 'train.csv'))
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    targets = df.target.values
    for fold, (train_index, test_index) in enumerate(kf.split(X=df, y=targets)):
        df.loc[test_index, 'kfold'] = fold
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)
