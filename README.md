# Introduction
This is an unofficial inplementation of [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf) in PyTorch.
This project is based on the work [here](https://github.com/amdegroot/ssd.pytorch#datasets). Thanks to [@amdegroot](https://github.com/amdegroot).
Currently, only support PASCAL VOC 2007 dataset.

# Dependencies
- `CUDA 10`
- `python3.5+`
- `Pytorch` (tested on 1.0.0)
- `Visdom` (for visualization training process)

# Dataset
### PASCAL VOC 2007
Run the command and it will download and unzip the dataset into `'./data'` folder.
```
$ bash scripts/voc2007.sh
```

# Train
- Download the fc-reduced VGG-16 PyTorch base network weights [here](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth):
```
$ mkdir weights
$ cd weights
$ wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```
- Run the following bash:
```
bash train.sh
```