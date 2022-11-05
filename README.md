# FAFS:  Fine-grained Adversarial training with Fourier transform and improved Self-learning

This is a [pytorch](http://pytorch.org/) implementation of [FAFS]().

## Set up
### 1. Prerequisites

### Create enviroment

```bash
# install python packages
pip install -r  requirements.txt

# install apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### Download datasets

**Source domain**

- [GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- [SYNTHIA Dataset]( http://synthia-dataset.net/download/808/ )

**Target domain**
- [Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

#### Download pretrained model
- Resnet101 pretrained models are from [Pytorch](https://pytorch.org/vision/stable/models.html)
- Mix-Transformer (MiT) pretrained models are downloaded from [SegFormer](https://github.com/NVlabs/SegFormer) (
[google drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) | 
[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ)
) and saved to `pretrained/`

### 2. Train
Our experiment is implemented in Pytorch on a single Nvidia GeForce RTX 3090. The training process is divided into three phases: source (`batch_size=8`), warm-up (`batch_size=4`) and self-training (3 times with `batch_size=8`)
```
cd code
bash run.sh
```

### 3. Evaluate
```
cd code
bash test.sh
```


## Visualization results

![Visualization](figures/visualization.png)

## Acknowledge
Main codes are adapted from [IAST](https://github.com/Raykoooo/IAST). We thank them for their excellent projects.

<!-- ### Citation
If you find this code useful please consider citing -->
