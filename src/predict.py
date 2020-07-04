import torch 
import glob
import numpy as np
import argparse
from model_dispatcher import MODEL_DISPATCHER
from dataset import MelonamaDataset, MelonamaTTADataset
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
import torchvision
from torchvision.transforms import FiveCrop, ToTensor, Lambda, Normalize, TenCrop
from utils import scale_and_map_df, modify_model

logger = logging.getLogger(__name__)
tz = pytz.timezone('Australia/Sydney')
syd_now = datetime.now(tz)


def predict(args, test_loader, model):
    model.eval()
    final_predictions = []
    with torch.no_grad():
        tk0 = tqdm(test_loader, total=len(test_loader))
        for data in tk0:
            for key, value in data.items():
                data[key] = value.to(args.device)                
            if not args.tta:
                predictions, _ = model(**data, args=args)
                predictions = predictions.cpu()
            else: 
                bs, ncrops, c, h, w = data['image'].shape
                if args.use_metadata:
                    predictions, _ = model(image=data['image'].view(-1, c, h, w), target=data['target'].view(-1), 
                        args=args, meta=data['meta'])
                else:
                    predictions, _ = model(image=data['image'].view(-1, c, h, w), target=data['target'].view(-1), 
                        args=args)
                predictions = predictions.view(bs, ncrops, -1).mean(1) 
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
    parser.add_argument('--tta', action='store_true', default=False, help="Test batch size.")
    parser.add_argument('--sz', type=int, default=292, help="Test batch size.")
    parser.add_argument('--loss', default='weighted_focal_loss', help="Loss fn to use")
    parser.add_argument('--arch_name', default='efficientnet-b0', help="EfficientNet architecture to use.")
    parser.add_argument('--use_metadata', default=False, action='store_true', help="Whether to use metadata")
    parser.add_argument('--num_crops', default=10, type=int, help="number of crops to use during tta")

    args = parser.parse_args()

    if 'efficient_net' in args.model_name:
        model = MODEL_DISPATCHER[args.model_name](pretrained=False, arch_name=args.arch_name)

    if args.use_metadata:
        model = modify_model(model, args)

    # load weights
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)

    # test augmentations
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    test_aug = albumentations.Compose([
        albumentations.CenterCrop(args.sz, args.sz) if args.sz else albumentations.NoOp(),
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
    ]) if not args.tta else torchvision.transforms.Compose([
        TenCrop(args.sz) if args.num_crops==10 else FiveCrop(args.sz),
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
        Lambda(lambda crops: torch.stack([Normalize(mean=mean, std=std)(crop) for crop in crops]))
    ])
    print(f"\ntest augmentations: {test_aug}\n")

    # create test images and create dummy targets
    df_test = pd.read_csv("/home/ubuntu/repos/kaggle/melonama/data/test.csv")
    test_images = df_test.image_name.tolist()
    test_image_paths = [os.path.join(args.test_data_dir, image_name+'.jpg') for image_name in test_images]
    test_targets = np.zeros(len(test_image_paths))        

    meta_array=None
    if args.use_metadata:
        # create meta array
        sex_dummy_test = pd.get_dummies(df_test['sex'])[
            ['male', 'female']]
        site_dummy_test = pd.get_dummies(df_test['anatom_site_general_challenge'])[
                ['head/neck', 'lower extremity', 'oral/genital', 'palms/soles', 'torso','upper extremity']]
        assert max(df_test.age_approx)<100
        age_test = df_test.age_approx.fillna(-5)/100
        meta_array = pd.concat([sex_dummy_test, site_dummy_test, age_test], axis=1).values

    # create test dataset based on tta or not
    if args.tta: 
        test_dataset = MelonamaTTADataset(test_image_paths, test_aug, meta_array=meta_array, nc=args.num_crops)
    else:
        test_dataset = MelonamaDataset(test_image_paths, test_targets, test_aug, meta_array=meta_array)
    
    print(f"test dataset: {test_dataset.__class__.__name__}")
    
    # create test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle=False, num_workers=4)

    predictions = predict(args, test_loader, model)
    predictions = np.vstack((predictions)).ravel()
    np.save(f"{args.output_dir}/{args.model_path.split('/')[-1].strip('.bin')}.npy", predictions)
    print(f"Predictions saved at {args.output_dir}/{args.model_path.split('/')[-1].strip('.bin')}.npy")


if __name__ == '__main__':
    main()