import torch 
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
from pathlib import Path
import torch.nn as nn

tz = pytz.timezone('Australia/Sydney')
syd_now = datetime.now(tz)

def train_one_epoch(args, train_loader, model, optimizer, weights):
    if args.weighted_loss: weights = weights.to(args.device)
    losses = AverageMeter()
    model.train()
    if args.accumulation_steps > 1: 
        print(f"Due to gradient accumulation of {args.accumulation_steps} using global batch size of {args.accumulation_steps*train_loader.batch_size}")
        optimizer.zero_grad()
    tk0 = tqdm(train_loader, total=len(train_loader))
    for b_idx, data in enumerate(tk0):
        images = data['image']
        targets = data['target']
        images = images.to(args.device)
        targets = targets.to(args.device)
        if args.accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()
        _, loss = model(images=images, targets=targets, weights=weights, args=args)

        with torch.set_grad_enabled(True):
            loss.backward()
            if (b_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg)
    return losses.avg
        

def evaluate(args, valid_loader, model):
    losses = AverageMeter()
    final_preds = []
    model.eval()
    with torch.no_grad():
        tk0 = tqdm(valid_loader, total=len(valid_loader))
        for data in tk0:
            images  = data['image']
            targets = data['target']
            images  = images.to(args.device)
            targets = targets.to(args.device)
            preds, loss = model(images=images, targets=targets, args=args)
            losses.update(loss.item(), valid_loader.batch_size)
            preds = preds.cpu()
            final_preds.append(preds)
            tk0.set_postfix(loss=losses.avg)
    return final_preds, losses.avg
        

def main():
    parser = argparse.ArgumentParser()

    #TODO: Add paramaeters as arguments
    # Required paramaters
    parser.add_argument(
        "--device", 
        default=None, 
        type=str, 
        required=True, 
        help="device on which to run the training"
    )
    parser.add_argument(
        '--training_folds_csv', 
        default=None, 
        type=str, 
        required=True, 
        help="training file with Kfolds"
    )
    parser.add_argument(
        '--model_name', 
        default='se_resnext_50',
        type=str, 
        required=True, 
        help="Name selected in the list: " + f"{','.join(MODEL_DISPATCHER.keys())}"
    )
    parser.add_argument(
        '--train_data_dir', 
        required=True, 
        help="Path to train data files."
    )
    parser.add_argument(
        '--kfold', 
        required=True,
        type=int,  
        help="Fold for which to run training and validation."
    )
    #Other parameters
    parser.add_argument('--pretrained', default=None, type=str, help="Set to 'imagenet' to load pretrained weights.")
    parser.add_argument('--train_batch_size', default=64, type=int, help="Training batch size.")
    parser.add_argument('--valid_batch_size', default=32, type=int, help="Validation batch size.")
    parser.add_argument('--learning_rate', default=1e-4, type=float, help="Learning rate.")
    parser.add_argument('--epochs', default=3, type=int, help="Num epochs.")
    parser.add_argument('--accumulation_steps', default=1, type=int, help="Gradient accumulation steps.")
    parser.add_argument('--sz', default=None, type=int, help="The size to which RandomCrop and CenterCrop images.")
    parser.add_argument('--weighted_loss', default=False, action='store_true', help="Whether to have weighted loss or not.")
    parser.add_argument('--focal_loss', default=False, action='store_true', help="Whether to use focal loss or not.")
    parser.add_argument('--external_csv_path', default=False, type=str, help="External csv path with melonama image names.")
    parser.add_argument('--cc', default=False, action='store_true', help="Whether to use color constancy or not.")

    args = parser.parse_args()
    
    # if args.sz, then print message and convert to int
    if args.sz: 
        print(f"Images will be resized to {args.sz}")
        args.sz = int(args.sz)

    # get training and valid data    
    df = pd.read_csv(args.training_folds_csv)
    if args.external_csv_path: 
        print("External data at {} will be added to all training folds.".format(Path(args.external_csv_path).parent))
        df_external = pd.read_csv(args.external_csv_path)
    df_train = df.query(f"kfold != {args.kfold}").reset_index(drop=True)
    df_valid = df.query(f"kfold == {args.kfold}").reset_index(drop=True)
    print(f"For kfold {args.kfold}; train_df: {df_train.shape}, valid_df: {df_valid.shape}")

    # calculate weights for NN loss
    weights = len(df_train)/df_train.target.value_counts().values 
    class_weights = torch.FloatTensor(weights)
    if args.weighted_loss: 
        print(f"assigning weights {weights} to loss fn.")
    if args.focal_loss: 
        print("Focal loss will be used for training.")

    # create model
    model = MODEL_DISPATCHER[args.model_name](pretrained=args.pretrained)
    model = model.to(args.device)
  
    train_aug = albumentations.Compose([
        albumentations.RandomScale(0.1),
        albumentations.Rotate(15),
        albumentations.RandomBrightnessContrast(0.1, 0.1),
        albumentations.Flip(p=0.5),
        albumentations.IAAAffine(shear=0.1),
        albumentations.RandomCrop(args.sz, args.sz) if args.sz else albumentations.NoOp(),
        albumentations.Cutout(1, 16, 16), 
        albumentations.Normalize(always_apply=True),
    ])

    valid_aug = albumentations.Compose([
        albumentations.CenterCrop(args.sz, args.sz) if args.sz else albumentations.NoOp(),
        albumentations.Normalize(always_apply=True),
    ])

    print(f"Using train augmentations: {train_aug}")

    # get train and valid images & targets and add external data if required
    train_images = df_train.image_name.tolist()
    if args.external_csv_path:
        external_images = df_external.image.tolist()
        train_images = train_images+external_images
    train_image_paths = [os.path.join(args.train_data_dir, image_name+'.jpg') for image_name in train_images]
    train_targets = df_train.target if not args.external_csv_path else np.concatenate([df_train.target.values, np.ones(len(external_images))])

    assert len(train_images) == len(train_targets), "Length of train images {} doesnt match length of targets {}".format(len(train_images), len(train_targets))

    valid_images = df_valid.image_name.tolist()
    valid_image_paths = [os.path.join(args.train_data_dir, image_name+'.jpg') for image_name in valid_images]
    valid_targets = df_valid.target

    # create train and valid dataset, dont use color constancy as already preprocessed in directory
    train_dataset = MelonamaDataset(train_image_paths, train_targets, train_aug, cc=args.cc)
    valid_dataset = MelonamaDataset(valid_image_paths, valid_targets, valid_aug, cc=args.cc)

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=4)

    # create optimizer and scheduler for training 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, threshold=3e-4, mode='min', verbose=True
    )

    es = EarlyStopping(patience=6, mode='min')

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(args, train_loader, model, optimizer, weights=None if not args.weighted_loss else class_weights)
        preds, valid_loss = evaluate(args, valid_loader, model)
        predictions = np.vstack(preds).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        preds_df = pd.DataFrame({'predictions': predictions, 'targets': valid_targets, 'valid_image_paths': valid_image_paths})
        preds_df.to_csv("/home/ubuntu/repos/kaggle/melonama/data/output/valid_fold_{}.csv".format(args.kfold), index=False)
        print(f"Epoch: {epoch}, Train loss: {train_loss}, Valid loss: {valid_loss}, AUC: {auc}")
        scheduler.step(valid_loss)
        es(
            valid_loss, model, 
            model_path=f"/home/ubuntu/repos/kaggle/melonama/models/{syd_now.strftime(r'%d%m%y')}/model_fold_{args.kfold}_{args.sz}_{auc}.bin"
        )
        if es.early_stop:
            print("Early stopping!")
            break


if __name__=='__main__':
    main()



