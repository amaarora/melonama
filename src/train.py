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

def train_one_epoch(args, train_loader, model, optimizer):
    losses = AverageMeter()
    model.train()
    tk0 = tqdm(train_loader, total=len(train_loader))
    for data in tk0:
        images  = data['image']
        targets = data['target']
        images  = images.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()
        _, loss = model(images=images, targets=targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg)
    return losses.avg
        

def evaluate(args, valid_loader, model):
    losses = AverageMeter()
    final_preds = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(valid_loader, total=len(valid_loader)):
            images  = data['image']
            targets = data['target']
            images  = images.to(args.device)
            targets = targets.to(args.device)
            preds, loss = model(images=images, targets=targets)
            losses.update(loss.item(), valid_loader.batch_size)
            final_preds.append(preds.cpu())
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
        '--data_dir', 
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
    parser.add_argument('--learning_rate', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument('--epochs', default=3, type=int, help="Num epochs.")

    args = parser.parse_args()
    
    # get training and valid data    
    df = pd.read_csv(args.training_folds_csv)
    df_train = df.query(f"kfold != {args.kfold}").reset_index(drop=True)
    df_valid = df.query(f"kfold == {args.kfold}").reset_index(drop=True)

    # create model
    model = MODEL_DISPATCHER[args.model_name](pretrained=args.pretrained)
    model = model.to(args.device)

    # get augmentations
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    
    train_aug = albumentations.Compose([
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
    ])
    valid_aug = albumentations.Compose([
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
    ])
    
    # get train and valid images & targets
    train_images = df_train.image_name.tolist()
    train_image_paths = [os.path.join(args.data_dir, 'train224/'+image_name+'.jpg') for image_name in train_images]
    train_targets = df_train.target

    valid_images = df_valid.image_name.tolist()
    valid_image_paths = [os.path.join(args.data_dir, 'train224/'+image_name+'.jpg') for image_name in valid_images]
    valid_targets = df_valid.target

    # create train and valid dataset
    train_dataset = MelonamaDataset(train_image_paths, train_targets, train_aug)
    valid_dataset = MelonamaDataset(valid_image_paths, valid_targets, valid_aug)

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, num_workers=4)

    # create optimizer and scheduler for training 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, threshold=0.001, mode='max'
    )

    es = EarlyStopping(patience=5, mode='max')

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(args, train_loader, model, optimizer)
        preds, valid_loss = evaluate(args, valid_loader, model)
        predictions = np.vstack((preds)).ravel()
        auc = sklearn.metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch: {epoch}, Train loss: {train_loss}, Valid loss: {valid_loss}, AUC: {auc}")
        scheduler.step(auc)
        es(auc, model, model_path=f"model_fold_{args.fold}.bin")
        if es.early_stop:
            print("Early stopping!")
            break

if __name__=='__main__':
    main()



