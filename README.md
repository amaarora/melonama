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
