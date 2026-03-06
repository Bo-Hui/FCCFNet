# FCCF-Net: A Fine-grained Channel and Cross-encoder Fusion Network for Infrared Small Target Detection

# Installation
## Step 1: Create a conda environment
```
conda create --name FCCFNet
conda activate FCCFNet
```

## Step 2: Install the required libraries via requirements.txt
After navigating to the project directory, use the following command to install the required libraries for the project.
```
pip install -r requirements.txt
```

# Dataset Preparation
## File Structure
```
| -datasets
  | -SIRST
    | -idx_427
      | -trainval.txt
      | -test.txt
      | -train.txt
      ······
    | -images
          | -Misc_1.png
          ......
    | -masks
          | -Misc_1.png
          ......
  | -IRSTD-1K
  | -NUDT-SIRST
```
Note: It is worth noting that the test and train sets used in the NUDT-SIRST dataset are consistent with those in DNANet (https://github.com/YeRen123455/Infrared-Small-Target-Detection/tree/master/dataset/NUDT-SIRST/50_50).

## Datasets Link
- SIRST - https://github.com/YimianDai/open-acm
- IRSTD-1K - https://github.com/RuiZhang97/ISNet
- NUDT-SIRST - https://github.com/YeRen123455/Infrared-Small-Target-Detection/tree/master

## Custom Dataset
If you have your own dataset, please use the `get_mean_std` function in `utils/data_set_stats_output.py` to calculate the mean and standard deviation of your dataset and create a new Dataset object.

# Training & Test
## Training
```
python train.py --img_size 512 --batch_size 8 --epochs 600 --learning_rate 0.001 --dataset sirst #SIRST
python train.py --img_size 512 --batch_size 8 --epochs 600 --learning_rate 0.001 --dataset IRSTD-1k #IRSTD-1K
python train.py --img_size 512 --batch_size 8 --epochs 600 --learning_rate 0.001 --dataset NUDT-SIRST #NUDT-SIRST
```

## Test
```
python val.py --dataset sirst --checkpoint your_checkpoint_path
```

## Model check point
The trained model weights can be obtained from https://pan.baidu.com/s/1PPp2F7wBo_NslKjcUau7ww?pwd=49eh. Before testing, please ensure that your image size setting is consistent with that in the paper.
