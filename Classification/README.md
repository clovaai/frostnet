# StatAssist & GradBoost: Classification
StatAssist & GradBoost: A Study on Optimal INT8 Quantization-aware Training from Scratch

## Dataset
- CIFAR10
- CIFAR100
- Imagenet_1K (ILSVRC2015)

## Model
- AlexNet
- VGGNet
- ResNet
- ShuffleNetV2
- MobileNetV2
- MobileNetV3
- MobileNetV3_ReLU

## Run example

- Train
Modify setting/train.json before run.   
```shell
python train.py

```
- Test
Modify setting/evaluate.json before run.   
```shell
python evaluate.py

```

## Latency Check Results (CPU) on Imagenet_1K
* Measured using AMD Ryzen Threadripper 1950X 16-Core Processor

- fbgemm 

| Model   | Size (FP) | Size(quantized) | Latency(FP) | Latency(quantized)|
|:-------:|:---------:|:---------------:|------------:|------------------:|
|resnet18 | 46.81 MB | 11.78 MB | 390 ms  | 187 ms |
|shufflenet_v2_x0_5| 5.56 MB | 1.48 MB | 216 ms  | 128 ms |
|shufflenet_v2_x1_0| 9.24 MB | 2.44MB | 383 ms  | 355 ms |
|mobilenet_v2_ReLU6| 14.21 MB | 3.58 MB | 436 ms  | 278 ms |
|mobilenet_v2_ReLU| 14.21 MB | 3.58 MB | 562 ms  | 108 ms |
|mobilenet_v3_large| 22.07 MB | 5.81 MB | 645 ms  | 431 ms |
|mobilenet_v3_small| 10.91 MB | 2.92 MB | 304 ms  | 276 ms |
|mobilenet_v3_ReLU_large| 22.06 MB | 5.57 MB | 512 ms  | 232 ms |
|mobilenet_v3_ReLU_small| 10.90 MB | 2.77 MB | 262 ms  | 150 ms |

- qnnpack

| Model   | Size (FP) | Size(quantized) | Latency(FP) | Latency(quantized)|
|:-------:|:---------:|:---------------:|------------:|------------------:|
|resnet18 | 46.81 MB | 11.78 MB | 390 ms  | 191 ms |
|shufflenet_v2_x0_5| 5.56 MB | 1.48 MB | 216 ms  | 124 ms |
|shufflenet_v2_x1_0| 9.24 MB | 2.44MB | 383 ms  | 350 ms |
|mobilenet_v2_ReLU6| 14.21 MB | 3.58 MB | 436 ms  | 259 ms |
|mobilenet_v2_ReLU| 14.21 MB | 3.58 MB | 562 ms  | 97 ms |
|mobilenet_v3_large| 22.07 MB | 5.81 MB | 645 ms  | 422 ms |
|mobilenet_v3_small| 10.91 MB | 2.92 MB | 304 ms  | 272 ms |
|mobilenet_v3_ReLU_large| 22.06 MB | 5.57 MB | 512 ms  | 232 ms |
|mobilenet_v3_ReLU_small| 10.90 MB | 2.77 MB | 262 ms  | 151 ms |
