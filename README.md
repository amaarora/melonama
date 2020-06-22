This is the base repository for Melonama Kaggle competition. 

To resize images simply run: 
```
mkdir data/train224
mkdir data/test224
python resize_images.py --input_folder /home/ubuntu/repos/kaggle/melonama/data/jpeg/train --output_folder /home/ubuntu/repos/kaggle/melonama/data/jpeg/train_512/ --mantain_aspect_ratio --sz 600 
```

To train the model 
```
python train.py --model_name se_resnext_50     --device cuda     --training_folds_csv '/home/ubuntu/repos/kaggle/melonama/data/stratified_group_5_fold.csv' --train_data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg/train_256/' --kfold 0     --pretrained imagenet     --train_batch_size 64     --valid_batch_size 32     --learning_rate 4e-4     --epochs 100 --sz 224 --accumulation_steps 2 --external_csv_path /home/ubuntu/repos/kaggle/melonama/data/external/isic2019/external_melonama.csv 
```

To run predictions: (do this for every model to create a `np` array for each model passed through `model_path`)
```
 python predict.py --model_name efficient_net --model_path /home/ubuntu/repos/kaggle/melonama/models/230620/efficient_net_fold_0_672_0.9261787601864147.bin --test_data_dir /home/ubuntu/repos/kaggle/melonama/data/jpeg/test_768_cc/ --sz 672 --tta --test_batch_size 16  
```
The predictions are by default created at the path `/home/ubuntu/repos/kaggle/melonama/data/output` and overwrite the past ones. 


To create submission file: (reads each pred file created above and averages the predictions to create ensemble preds.)
```
python create_submission.py
```

To submit the `submission.csv` file to kaggle, simply run: 
```
kaggle competitions submit -c siim-isic-melanoma-classification -f submission.csv -m "Message"
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

## 16 Jun, 2019
- Gradient accumulations is used in this for training the model
- Spent the day trying to train the model on (512,512) images using `RandomResizeCrop` and `CenterCrop`. Original raw images were resized to mantain aspect ratio and shorter side to have 600px.
- Model training took too long on a single V100 and did not end up submitting to Kaggle.

```
python ../src/train.py --model_name se_resnext_50     \
--device cuda     \
--training_folds_csv \
'/home/ubuntu/repos/kaggle/melonama/data/train_folds.csv'     \
--data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg'     \
--kfold 1     \
--pretrained imagenet     \
--train_batch_size 16     \
--valid_batch_size 32     \
--learning_rate 1e-4     \
--epochs 50 \
--accumulation_steps 8
```

## 17 Jun, 2019 
- Created a new dataset, resized images by first adding Reflection padding using Fastai and then resizing to (224,224)
- Trained the model using the below commands, with gradient accumulations but the model does not perform well.
- Also tried this new dataset without Gradient Accumulation, did not work
- Also, read about **Temperature Scaling** which was interesting.
```
python train.py --model_name se_resnext_50     \
--device cuda     \
--training_folds_csv \
'/home/ubuntu/repos/kaggle/melonama/data/train_folds.csv'     \
--train_data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg/train_pad_224'     \
--kfold 0     \
--pretrained imagenet     \
--train_batch_size 64     \
--valid_batch_size 32     \
--learning_rate 1e-4     \
--epochs 50 \
--accumulation_steps 2 \
--sz 224
```

## 18 Jun, 2019 
- Since the model is not performing well for images resized with Reflection padding it is time to think of something different. 
- I do not intend to retrain the model on squished images, so created a new dataset that is resized by mantaining Aspect Ratio and shorter side has length 300px.
- Can try to add `GroupKFold` in a future iteration.

- The new baseline when I tried to replicate results using the training command shown below, and the logs are stored in 18 Jun 2020 part-2 notebook is auc 0.90 on the public test leaderboard.
```
python train.py --model_name se_resnext_50     \
--device cuda     \
--training_folds_csv \
'/home/ubuntu/repos/kaggle/melonama/data/train_folds.csv'     \
--train_data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg/train224/'     \
--kfold 1     \
--pretrained imagenet     \
--train_batch_size 64     \
--valid_batch_size 32     \
--learning_rate 1e-4     \
--epochs 100 
```

- Next currently, the model is training for (292,292) image size by randomly resizing (300px, X) sized images on train and using CenterCrop on the validation set. Also, we need to use gradient accumulation of 2. We get a public leaderboard score of 0.901.
```
python train.py --model_name se_resnext_50     \
--device cuda     \
--training_folds_csv \
'/home/ubuntu/repos/kaggle/melonama/data/train_folds.csv'     \
--train_data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg/train_300px_ar/'     \
--kfold 2     \
--pretrained imagenet     \
--train_batch_size 32     \
--valid_batch_size 32     \
--learning_rate 1e-4     \
--epochs 100 \
--sz 292 \
--accumulation_steps 2
```

- I have also added TTA, everything else same as above, but this did not improve the public leaderboard score and got a score of 0.897.

- Now, model is training for weighted BCE with everything same as above.
```
python train.py --model_name se_resnext_50     \
--device cuda     \
--training_folds_csv \
'/home/ubuntu/repos/kaggle/melonama/data/train_folds.csv'     \
--train_data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg/train_300px_ar/'     \
--kfold 0     \
--pretrained imagenet     \
--train_batch_size 64     \
--valid_batch_size 32     \
--learning_rate 1e-3     \
--epochs 100 \
--sz 224 \
--accumulation_steps 4 \
--weighted_loss
```

## 20 Jun, 2019
- External data training is added (only melonama images from ISIC 2019 have been added to help with imbalance)
- AUC score stuck at 0.90 (this is the best I can do so far)
- I think I waste the whole day trying to train models, and just not enough on how can I actually increase scores. There is a new idea to ensemble models with different sizes to get higher LB score. 
- Added a `/home/ubuntu/repos/kaggle/melonama/data/stratified_group_5_fold.csv` csv file which splits the train and validation data based on `patient_id` as well and makes sure there is no overlap. 
- Script for `folds.py` has also been updated to include the new code. Usage instructions are included in the `stratified_group_k_fold` function.
- Added a new data folder `/home/ubuntu/repos/kaggle/melonama/data/jpeg/train_512/` which contains train images that have been resized to 600 x 600 and should be used for training models with image size 512x512 random and center crop.
- Read about AUC to get a better understanding of what it is. 
- Added and tried using focal loss. 
- Also new validation csv is spit out as well that contains validation targets and predictions for analysis.
- Perhaps more effort needs to be put into rewriting code and updating the model.
- Also, added `EfficientNet`
```
python train.py --model_name se_resnext_50     --device cuda     --training_folds_csv '/home/ubuntu/repos/kaggle/melonama/data/stratified_group_5_fold.csv' --train_data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg/train_256/' --kfold 0     --pretrained imagenet     --train_batch_size 64     --valid_batch_size 32     --learning_rate 4e-4     --epochs 100 --sz 224 --accumulation_steps 2 --external_csv_path /home/ubuntu/repos/kaggle/melonama/data/external/isic2019/external_melonama.csv
```

## 21 Jun, 2019 
- Add shades of gray color constancy algorithm from [here](https://github.com/nickshawn/Shades_of_Gray-color_constancy_transformation/blob/master/color_constancy.py).
- Update dataset and add color_constancy as preprocessing step
- Update augmentation to remove replace `RandomResizeCrop` with `RandomCrop`. This is from 2019 winner solution. 
- Update data augmentation and use `valid_loss` for scheduler step and early stopping.