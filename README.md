# SIIM-ISIC Melanoma Classification - top 5% solution
Public Leaderboard: 773/3314 | Private Leaderboard: 153/3314


This repository contains the codebase for my approach on recent [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview) kaggle competition. 

A complete summary and walkthrough can be find in my blog post [here]().

## Data
To be able to replicate the experiments and re run the code, you will need to download the data from [here](https://www.kaggle.com/c/siim-isic-melanoma-classification/data).

## Training Folds
To create folds, simply run, `python folds.py` or use the triple stratified training folds file that is part of this repo. This will split the training data into 5 folds. 

## Image Preprocessing
To preprocess and resize images, simply run: 
`python resize_images.py --input_folder <path_to_input_folder> --output_folder <path_to_output_folder> --mantain_aspect_ratio --sz 224`

This will resize all images such that the shorter side of the image is of size 224px while mantaining the aspect ratio of the image. This kind of preprocessing was also used by ISIC 2019 winners as in the research paper [here](https://isic-challenge-stade.s3.amazonaws.com/99bdfa5c-4b6b-4c3c-94c0-f614e6a05bc4/method_description.pdf?AWSAccessKeyId=AKIA2FPBP3II4S6KTWEU&Signature=nXwY%2BI7mHPE0Nf%2BhY14z4PpajSU%3D&Expires=1598164223).

To also add color constancy, simply add `-- cc` flag to the command. 

For a complete list of parameters, run `python resize_images.py -h` in the `src` directory of this GitHub repo. 

## Model Training

To train the model, run the following command:  

```
python train.py --model_name efficient_net \
--arch_name efficientnet-b3 --device cuda --metric 'auc' \
--training_folds_csv <path_to_training_folds_csv> \
--train_data_dir <path_to_preprocessed_data> \
--kfold 0,1,2,3,4 --pretrained imagenet --train_batch_size 64 \
--valid_batch_size 64 --learning_rate  5e-4 \
--epochs 100 --sz 384 --accumulation_steps 8 \
--loss 'weighted_focal_loss'  
```

## Inference
To run inference, 

```
python predict.py --model_name efficient_net --arch_name efficientnet-b3 \
--model_path <path_to_trained_model_output_from_training_script> \
--test_data_dir <path_to_preprocessed_test_data> --sz 256 --test_batch_size 64 
```

To predict using TTA, just add `--tta` flag to the above command.