import torch 
import glob
import numpy as np
import argparse
from model_dispatcher import MODEL_DISPATCHER
from dataset import MelonamaDataset
import pandas as pd
import albumentations
from early_stopping import EarlyStopping
from tqdm import tqdm
from average_meter import AverageMeter
import sklearn
import os
from sklearn import metrics
from datetime import date, datetime
import pytz
import logging

logger = logging.getLogger(__name__)
tz = pytz.timezone('Australia/Sydney')
syd_now = datetime.now(tz)


def predict(args, test_loader, model):
    model.eval()
    final_predictions = []
    with torch.no_grad():
        tk0 = tqdm(test_loader, total=len(test_loader))
        for data in tk0:
            images  = data['image']
            targets = data['target']
            images  = images.to(args.device)    
            targets = targets.to(args.device)                
            predictions, _ = model(images=images, targets=targets)
            predictions = predictions.cpu()
            final_predictions.append(predictions)
    return final_predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", 
        default=None, 
        type=str, 
        required=True, 
        help="Model type on which to run predictions."
    )
    parser.add_argument(
        "--model_path", 
        default=None, 
        type=str, 
        required=True, 
        help="Model to use to make predictions."
    )
    parser.add_argument(
        "--test_data_dir", 
        default=None, 
        type=str, 
        required=True, 
        help="Directory where test data files exist."
    )
    #Other parameters
    parser.add_argument('--device', default='cuda', type=str, help="Device on which to run predictions.")
    parser.add_argument('--test_batch_size', default=64, type=int, help="Test batch size.")
    parser.add_argument('--submission_file', default="/home/ubuntu/repos/kaggle/melonama/data/sample_submission.csv", type=str, help="Test batch size.")
    parser.add_argument('--output_dir', default="/home/ubuntu/repos/kaggle/melonama/data/output", type=str, help="Test batch size.")
    
    args = parser.parse_args()

    model = MODEL_DISPATCHER[args.model_name](pretrained=None)
    
    # load weights
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)

    # test augmentations
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    test_aug = albumentations.Compose([
        # albumentations.CenterCrop(224, 224),
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
    ])

    # create test images and create dummy targets
    df_test = pd.read_csv("/home/ubuntu/repos/kaggle/melonama/data/test.csv")
    test_images = df_test.image_name.tolist()
    test_image_paths = [os.path.join(args.test_data_dir, image_name+'.jpg') for image_name in test_images]
    test_targets = np.zeros(len(test_image_paths))        
    
    # create test dataset
    test_dataset = MelonamaDataset(test_image_paths, test_targets, test_aug)

    # create test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
        shuffle=False, num_workers=4)

    predictions = predict(args, test_loader, model)
    predictions = np.vstack((predictions)).ravel()
    np.save(f"{args.output_dir}/{args.model_path.split('/')[-1].strip('.bin')}.npy", predictions)
    print(f"Predictions saved at {args.output_dir}/{args.model_path.split('/')[-1].strip('.bin')}.npy")


if __name__ == '__main__':
    main()