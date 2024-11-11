# DGUNET
This repo holds the official code for work "A Double Gate UNet for Coarse to Fine Coronary Artery Segmentation"

## 1. Dependencies and Installation
* Clone this repo:
```
https://github.com/just12321/DGUNET.git
cd DGUNET
```
* Create a conda virtual environment and activate:
```
conda create -n dgunet python=3.10 -y
conda activate dgunet
```
* Install packages:
```
pip install -r requirements.txt
```
## 2. Data Preparation
The directory structure of the whole project is as follows:
```
.
├── dataset
│   ├── Database134
│   │   ├── train
│   │   │   ├── images
│   │   │   └── manual
│   │   └── val
│   │       ├── images
│   │       └── manual
│   ├── DatasetCHUAC
│   │   └── train
│   │       ├── images
│   │       └── manual
│   ├── Custom
│   │      └──
├── model
│   └──
├── utils
│   └── 
└── main.py
```
## 3. Training
This is not very convenient for running custom configurations, which requires modifying the main.py code itself. We will refactor and optimize this in the near future.
- Run the train script.
```
python train.py
```

