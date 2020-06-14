import numpy as np 
import pandas as pd 
import glob

if __name__ == '__main__':
    sub = pd.read_csv('/home/ubuntu/repos/kaggle/melonama/data/sample_submission.csv')
    np_array_paths = glob.glob("/home/ubuntu/repos/kaggle/melonama/data/output/*.npy")
    predictions = [np.load(path) for path in np_array_paths]
    predictions = sum(predictions) / len(predictions)
    sub['target'] = predictions
    sub.to_csv("/home/ubuntu/repos/kaggle/melonama/data/output/submission.csv", index=False)