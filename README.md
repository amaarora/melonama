This is the base repository for Melonama Kaggle competition. 

To resize images simply run: 
```
mkdir data/train224
mkdir data/test224
python resize_images.py --input_folder /home/ubuntu/repos/kaggle/melonama/data/jpeg/train --output_folder /home/ubuntu/repos/kaggle/melonama/data/jpeg/train224ar --mantain_aspect_ratio True
```

To train the model 
```
python train.py --model_name se_resnext_50 \
    --device cuda \
    --training_folds_csv '/home/ubuntu/repos/kaggle/melonama/data/train_folds.csv' \
    --data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg' \
    --kfold 0 \
    --pretrained imagenet \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 50 
```

To run predictions: (do this for every model to create a `np` array for each model passed through `model_path`)
```
python predict.py --model_name se_resnext_50 \
    --model_path /home/ubuntu/repos/kaggle/melonama/models/140620/model_fold_0.bin \
    --test_data_dir /home/ubuntu/repos/kaggle/melonama/data/jpeg/test224 \
```

To create submission file: (reads each pred file created above and averages the predictions to create ensemble preds.)
```
python create_submission.py
```

# Training History and Improvements 

## 14 Jun, 2019 
- Trained `SeResnext_50x4d` for Images using simple resize to 224x224. 
- five fold ensemble has public test set leaderboard accuracy of 0.911
- Training command used: 
```
python train.py --model_name se_resnext_50 \
    --device cuda \
    --training_folds_csv '/home/ubuntu/repos/kaggle/melonama/data/train_folds.csv' \
    --data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg' \
    --kfold 0 \
    --pretrained imagenet \
    --train_batch_size 64 \
    --valid_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 50 
```
- Uses train and validation batch_size of 64

## 15 Jun, 2019
- Found multiple research papers and past year approaches [here](https://challenge2019.isic-archive.com/leaderboard.html).
- Following the preprocessing steps of Dysion AI available [here](https://isic-challenge-stade.s3.amazonaws.com/9e2e7c9c-480c-48dc-a452-c1dd577cc2b2/ISIC2019-paper-0816.pdf?AWSAccessKeyId=AKIA2FPBP3II4S6KTWEU&Signature=XpwPLQaZ6JP3gekmjmhXfRTldIg%3D&Expires=1592248281), I updated the preprocess step and created two more two data folders `train224ar` and `test224ar` which preprocess the images and resize all images keeping the aspect ratio and the shorter length of the resized image is 600px. 
- Added `albumentations.RandomResizeCrop` data augmentation and resized images to 512x512 in training set, while used `albumentation.CenterCrop` in Validation images.

- Model takes around 10 minutes to train which is twice the time taken to train the 224x224 images in the first iteration. 
- Also, updated the code to include gradient accumulation which can be used by passing the `args.accumulation_steps`. 
- This iteration uses train `batch_size` of 16 while it uses validation `batch_size` of 32. 
- Gradient accumulation is not used in training for this iteration 
- Trainig command used:
``` 
python train.py --model_name se_resnext_50 \
    --device cuda \
    --training_folds_csv '/home/ubuntu/repos/kaggle/melonama/data/train_folds.csv' \
    --data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg' \
    --kfold 0 \
    --pretrained imagenet \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 50 
```