This is the base repository for Melonama Kaggle competition. 

To resize images simply run: 
```
mkdir data/train224
mkdir data/test224
python resize_images.py # in src dir
```

To train the model 
```
python train.py --model_name se_resnext_50 \
    --device cuda \
    --training_folds_csv '/home/ubuntu/repos/kaggle/melonama/data/train_folds.csv' \
    --data_dir '/home/ubuntu/repos/kaggle/melonama/data/jpeg' \
    --kfold 0 \
    --pretrained imagenet \
    --train_batch_size 64 \
    --valid_batch_size 32 \
    --learning_rate 1e-3 \
    --epochs 1 
```

To run predictions:
```
python predict.py --model_name se_resnext_50 \
    --model_path /home/ubuntu/repos/kaggle/melonama/models/140620/model_fold_0.bin \
    --test_data_dir /home/ubuntu/repos/kaggle/melonama/data/jpeg/test224 \
```
