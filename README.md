## FrostNet: Towards Quantization-Aware Network Architecture Search

**Taehoon Kim<sup>1,2</sup>, YoungJoon Yoo<sup>1</sup>, Jihoon Yang<sup>2</sup>** | [Paper](https://arxiv.org/pdf/2007.00992.pdf) | [Pretrained Models](#pretrained)


1. Clova AI Research, NAVER Corp.
2. Sogang University Machine Learning Lab.


## Abstract

INT8 quantization has become one of the standard techniques for deploying convolutional neural networks (CNNs) on edge devices to reduce the memory and computational resource usages. By analyzing quantized performances of existing mobile-target network architectures, we can raise an issue regarding the importance of network architecture for optimal INT8 quantization. In this paper, we present a new network architecture search (NAS) procedure to find a network that guarantees both full-precision (FLOAT32) and quantized (INT8) performances. We first propose critical but straightforward optimization method which enables quantization-aware training (QAT) : floating-point statistic assisting (StatAssist) and stochastic gradient boosting (GradBoost). By integrating the gradient-based NAS with StatAssist and GradBoost, we discovered a quantization-efficient network building block, Frost bottleneck. Furthermore, we used Frost bottleneck as the building block for hardware-aware NAS to obtain quantization-efficient networks, FrostNets, which show improved quantization performances compared to other mobile-target networks while maintaining competitive FLOAT32 performance. Our FrostNets achieve higher recognition accuracy than existing CNNs with comparable latency when quantized, due to higher latency reduction rate (average 65%).


## Model performances
### ImageNet classification results

- Accuracy comparison with other state of the art lightweight models:

  <img src=etc/acc_latency.png width=480 hspace=30><br>
  <img src=etc/classification.png width=720> 

### COCO detection results
- mAP scores comparison on MS COCO val split 2017 with RetinaNet and Faster-RCNN:

  <img src=etc/detection_retina.png width=360 hspace=30> <img src=etc/detection_faster.png width=360>


<h2 id="pretrained"> Pretrained models </h2>

- We provide FrostNets' pretrained weights on ImageNet dataset. Note that all the models are trained and evaluated with 224x224 image size. [Google Drive](https://drive.google.com/file/d/196nKcns-6f1drrcl1mpD1MAIQxXCxyhF/view?usp=sharing)
   
## Getting Started


### Training your own FrostNet
We trained FrostNets with one of the popular imagenet classification code, rwightman's [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) for more efficient training. After including FrostNet's model file into the training code, one can train FrostNets with the command line in [training_confs](./training_commands.txt).

### Post quantization examples
We also provide post-quantization supported version of rwightman's pytorch-image-models in [quanitzation-pytorch-image-models](https://github.com/tgisaturday/pytorch-image-models) for easier post-quantization with PyTorch.

### Training object detection models with FrostNet backbones
We trained FrostNets with one of the popular object detection project, [mmdetection](https://github.com/rwightman/pytorch-image-models) for more efficient training. Include [frostnet_features.py](./frostnet_features.py) to mmdetection codes to train models. 


### StatAssist & Gradboost examples

#### Supports

- Classification (AlexNet, VGG, Resnet, ShuffleNetV2, Mobilenet V2 & V3) [(details)](./Classification/README.md)
- Object Detection (TDSOD, SSDLITE-MobileNet V2) [(details)](./Object_Detection/README.md)
- Semantic Segmentation (ESPNet V1 & V2, Mobilenet V2 & V3) [(details)](./Semantic_Segmentation/README.md)
- Style Transfer (Pix2Pix, CycleGAN) [(details)](./Style_Transfer/README.md)

#### Implementations

- Our StatAssist implementations can be found in:
  - Classification: line 149 - 164 in [here](./Classification/train.py).
  - Object Detection: line 185 - 239 in [here](./Object_Detection/qtrainval.py).
  - Semantic Segmentation: line 205 - 221 in [here](./Semantic_Segmentation/train.py).
  - Style Transfer: line 42 - 64 in [here](./Style_Transfer/train.py).
 
- Our GradBoost version of optimizers can be found [here](./optimizer.py). 

## Update
- November 27th, 2020
  - FrostNet, quantization-aware neural network architecture, updated. [(details)](./frostnet.py)
- July 29th, 2020
  - Quantized CPU latency results updated. [(details)](./Classification/README.md)
  
 ## License

This project is distributed under MIT license.

```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## How to cite

```
@misc{kim2020statassist,
    title={StatAssist & GradBoost: A Study on Optimal INT8 Quantization-aware Training from Scratch},
    author={Taehoon Kim and Youngjoon Yoo and Jihoon Yang},
    year={2020},
    eprint={2006.09679},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
