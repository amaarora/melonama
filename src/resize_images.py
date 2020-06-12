import os
import glob
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed

def resize_and_save(path, output_path, sz: tuple):
    fn = os.path.basename(path)  
    im = Image.open(path)
    im = im.resize(sz, resample=Image.BILINEAR)
    im.save(os.path.join(output_path, fn))

if __name__ == '__main__': 
    input_folder  = "/home/ubuntu/repos/kaggle/melonama/data/jpeg/test/"
    output_folder = "/home/ubuntu/repos/kaggle/melonama/data/jpeg/test224/"
    images = glob.glob(os.path.join(input_folder, '*.jpg'))
    Parallel(n_jobs=16)(
        delayed(resize_and_save)(i, output_folder, (224, 224)) for i in tqdm(images))