# OctConv-TFKeras
Unofficial implementation of Octave Convolutions (OctConv) in TensorFlow / Keras. 

Y. Chen, H. Fang, B. Xu, Z. Yan, Y. Kalantidis, M. Rohrbach, S. Yan, J. Feng. *Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution*. (2019). https://arxiv.org/abs/1904.05049

![](octconv_02.png)

(Update 2019-04-26) Official implementation by MXNet is available : [https://github.com/facebookresearch/OctConv](https://github.com/facebookresearch/OctConv) 

# Usage
```python
from oct_conv2d import OctConv2D
# high, low = some tensors or inputs
high, low = OctConv2D(filters=ch, alpha=alpha)([high, low])
```

# Colab Notebook
**Train OctConv ResNet** (TPU)  
https://colab.research.google.com/drive/1MXN46mhCk6s-G_nfJrH1B6_8GXh-a_QH

**Measuring prediction time** (CPU)  
https://colab.research.google.com/drive/12MdVXyB9K3FnpzYNmyc3qu5s59-53WNE

# CIFAR-10
Experimented with Wide ResNet (N = 4, k = 10). Train with colab TPUs.

| alpha | Test Accuracy |
|:-----:|:----------:|
|   0   |   88.47%   |
|  0.25 |   94.83%   |
|  0.5  |   94.40%   |
|  0.75 |   93.54%   |

![](figs/training_curve.png)

# Prediction Time
CPU and GPU are colab environment. On CPU, use 256 samples for prediction, and on GPU, use 50000 samples for prediction. Both CPU and GPU are 32x32 resolution each.

## CPU
| alpha/s |  Mean | Median |   S.D. | Median/sample[ms] | Relative measured value | Theoretical FLOPs cost |
|:-------:|------:|-------:|-------:|------------------:|------------------------:|-----------------------:|
|    0    | 47.00 |  46.94 | 0.3755 |            183.36 |                     100 |                    100 |
|   0.25  | 35.23 |  35.06 | 0.7022 |            136.95 |                      75 |                     67 |
|   0.5   | 24.22 |  24.04 | 0.8978 |             93.91 |                      51 |                     44 |
|   0.75  | 16.92 |  16.67 | 1.1320 |             65.12 |                      36 |                     30 |

Theoretical FLOPs cost are from the paper.

## GPU
| alpha/s |  Mean | Median |   S.D. | Median/sample[ms] | Relative measured value |
|:-------:|------:|-------:|-------:|------------------:|------------------------:|
|    0    | 72.27 |  72.47 | 0.8495 |              1.45 |                     100 |
|   0.25  | 74.92 |  74.78 | 0.7479 |              1.50 |                     103 |
|   0.5   | 57.16 |  56.99 | 0.8042 |              1.14 |                      79 |
|   0.75  | 41.06 |  40.84 | 1.0120 |              0.82 |                      56 |

You can see that the time spent on prediction decreases with theoretical FLOPs costs, both CPU and GPU.

# Details (Japanese)
[最新論文]Octave Convolution（OctConv）を試してみる  
[https://qiita.com/koshian2/items/0e40a5930f1aa63a66b9](https://qiita.com/koshian2/items/0e40a5930f1aa63a66b9)
