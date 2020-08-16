# StatAssist & GradBoost

Official PyTorch implementation of [StatAssist & GradBoost: A Study on Optimal INT8 Quantization-aware Training from Scratch](https://arxiv.org/abs/2006.09679) 

Taehoon Kim<sup>1,2</sup>, YoungJoon Yoo<sup>1</sup>, Jihoon Yang<sup>2</sup><br>

1. Clova AI Research, NAVER Corp.
2. Sogang University Machine Learning Lab.

### Abstract

This paper studies the scratch training of quantization-aware training (QAT), which has been applied to the lossless conversion of lower-bit, especially for INT8 quantization. Due to its training instability, QAT have required a full-precision (FP) pre-trained weight for fine-tuning and the performance is bound to the original FP model with floating-point computations. Here, we propose critical but straightforward optimization methods which enable the scratch training: floating-point statistic assisting (StatAssist) and stochastic-gradient boosting (GradBoost). We discovered that, first, the scratch QAT get comparable and often surpasses the performance of the floating-point counterpart without any help of the pre-trained model, especially when the model becomes complicated.We also show that our method can even train the minimax generation loss, which is very unstable and hence difficult to apply QAT fine-tuning. From extent experiments, we show that our method successfully enables QAT to train various deep models from scratch: classification, object detection, semantic segmentation, and style transfer, with comparable or often better performance than their FP baselines.


## Requirements

- python 3
- pytorch >= 1.4.0
- torchvision >= 0.5.0
- opencv-python
- numpy
- pillow
- tqdm
- visdom

## Supports

- Classification (AlexNet, VGG, Resnet, ShuffleNetV2, Mobilenet V2 & V3) [(details)](./Classification/README.md)
- Object Detection (TDSOD, SSDLITE-MobileNet V2) [(details)](./Object_Detection/README.md)
- Semantic Segmentation (ESPNet V1 & V2, Mobilenet V2 & V3) [(details)](./Semantic_Segmentation/README.md)
- Style Transfer (Pix2Pix, CycleGAN) [(details)](./Style_Transfer/README.md)

## Implementations

- Our StatAssist implementations can be found in:
  - Classification: line 149 - 164 in [here](./Classification/train.py).
  - Object Detection: line 185 - 239 in [here](./Object_Detection/qtrainval.py).
  - Semantic Segmentation: line 205 - 221 in [here](./Semantic_Segmentation/train.py).
  - Style Transfer: line 42 - 64 in [here](./Style_Transfer/train.py).
 
- Our GradBoost version of optimizers can be found [here](./optimizer.py). 

## Update

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
