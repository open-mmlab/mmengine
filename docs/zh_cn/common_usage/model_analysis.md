# 统计模型计算量和参数量

1. 定义模型

   ```python
   import torch.nn.functional as F
   import torchvision
   from mmengine.model import BaseModel

   class MMResNet50(BaseModel):
       def __init__(self):
           super().__init__()
           self.resnet = torchvision.models.resnet50()

       def forward(self, imgs, labels=None, mode='tensor'):
           x = self.resnet(imgs)
           if mode == 'loss':
               return {'loss': F.cross_entropy(x, labels)}
           elif mode == 'predict':
               return x, labels
           elif mode == 'tensor':
               return x
   ```

2. 统计计算量和参数量

   ```python
   from mmengine.analysis import get_model_complexity_info

   input_shape = (3, 224, 224)
   model = MMResNet50()
   analysis_results = get_model_complexity_info(model, input_shape)
   ```

- 以表格的形式显示

  ```python
  print(analysis_results['out_table'])
  ```

  <details>
    <summary>点击展开</summary>

  ```html
  +------------------------+----------------------+------------+--------------+
  | module                 | #parameters or shape | #flops     | #activations |
  +------------------------+----------------------+------------+--------------+
  | resnet                 | 25.557M              | 4.145G     | 11.115M      |
  |  conv1                 |  9.408K              |  0.118G    |  0.803M      |
  |   conv1.weight         |   (64, 3, 7, 7)      |            |              |
  |  bn1                   |  0.128K              |  4.014M    |  0           |
  |   bn1.weight           |   (64,)              |            |              |
  |   bn1.bias             |   (64,)              |            |              |
  |  layer1                |  0.216M              |  0.69G     |  4.415M      |
  |   layer1.0             |   75.008K            |   0.241G   |   2.007M     |
  |    layer1.0.conv1      |    4.096K            |    12.845M |    0.201M    |
  |    layer1.0.bn1        |    0.128K            |    1.004M  |    0         |
  |    layer1.0.conv2      |    36.864K           |    0.116G  |    0.201M    |
  |    layer1.0.bn2        |    0.128K            |    1.004M  |    0         |
  |    layer1.0.conv3      |    16.384K           |    51.38M  |    0.803M    |
  |    layer1.0.bn3        |    0.512K            |    4.014M  |    0         |
  |    layer1.0.downsample |    16.896K           |    55.394M |    0.803M    |
  |   layer1.1             |   70.4K              |   0.224G   |   1.204M     |
  |    layer1.1.conv1      |    16.384K           |    51.38M  |    0.201M    |
  |    layer1.1.bn1        |    0.128K            |    1.004M  |    0         |
  |    layer1.1.conv2      |    36.864K           |    0.116G  |    0.201M    |
  |    layer1.1.bn2        |    0.128K            |    1.004M  |    0         |
  |    layer1.1.conv3      |    16.384K           |    51.38M  |    0.803M    |
  |    layer1.1.bn3        |    0.512K            |    4.014M  |    0         |
  |   layer1.2             |   70.4K              |   0.224G   |   1.204M     |
  |    layer1.2.conv1      |    16.384K           |    51.38M  |    0.201M    |
  |    layer1.2.bn1        |    0.128K            |    1.004M  |    0         |
  |    layer1.2.conv2      |    36.864K           |    0.116G  |    0.201M    |
  |    layer1.2.bn2        |    0.128K            |    1.004M  |    0         |
  |    layer1.2.conv3      |    16.384K           |    51.38M  |    0.803M    |
  |    layer1.2.bn3        |    0.512K            |    4.014M  |    0         |
  |  layer2                |  1.22M               |  1.043G    |  3.111M      |
  |   layer2.0             |   0.379M             |   0.379G   |   1.305M     |
  |    layer2.0.conv1      |    32.768K           |    0.103G  |    0.401M    |
  |    layer2.0.bn1        |    0.256K            |    2.007M  |    0         |
  |    layer2.0.conv2      |    0.147M            |    0.116G  |    0.1M      |
  |    layer2.0.bn2        |    0.256K            |    0.502M  |    0         |
  |    layer2.0.conv3      |    65.536K           |    51.38M  |    0.401M    |
  |    layer2.0.bn3        |    1.024K            |    2.007M  |    0         |
  |    layer2.0.downsample |    0.132M            |    0.105G  |    0.401M    |
  |   layer2.1             |   0.28M              |   0.221G   |   0.602M     |
  |    layer2.1.conv1      |    65.536K           |    51.38M  |    0.1M      |
  |    layer2.1.bn1        |    0.256K            |    0.502M  |    0         |
  |    layer2.1.conv2      |    0.147M            |    0.116G  |    0.1M      |
  |    layer2.1.bn2        |    0.256K            |    0.502M  |    0         |
  |    layer2.1.conv3      |    65.536K           |    51.38M  |    0.401M    |
  |    layer2.1.bn3        |    1.024K            |    2.007M  |    0         |
  |   layer2.2             |   0.28M              |   0.221G   |   0.602M     |
  |    layer2.2.conv1      |    65.536K           |    51.38M  |    0.1M      |
  |    layer2.2.bn1        |    0.256K            |    0.502M  |    0         |
  |    layer2.2.conv2      |    0.147M            |    0.116G  |    0.1M      |
  |    layer2.2.bn2        |    0.256K            |    0.502M  |    0         |
  |    layer2.2.conv3      |    65.536K           |    51.38M  |    0.401M    |
  |    layer2.2.bn3        |    1.024K            |    2.007M  |    0         |
  |   layer2.3             |   0.28M              |   0.221G   |   0.602M     |
  |    layer2.3.conv1      |    65.536K           |    51.38M  |    0.1M      |
  |    layer2.3.bn1        |    0.256K            |    0.502M  |    0         |
  |    layer2.3.conv2      |    0.147M            |    0.116G  |    0.1M      |
  |    layer2.3.bn2        |    0.256K            |    0.502M  |    0         |
  |    layer2.3.conv3      |    65.536K           |    51.38M  |    0.401M    |
  |    layer2.3.bn3        |    1.024K            |    2.007M  |    0         |
  |  layer3                |  7.098M              |  1.475G    |  2.158M      |
  |   layer3.0             |   1.512M             |   0.376G   |   0.652M     |
  |    layer3.0.conv1      |    0.131M            |    0.103G  |    0.201M    |
  |    layer3.0.bn1        |    0.512K            |    1.004M  |    0         |
  |    layer3.0.conv2      |    0.59M             |    0.116G  |    50.176K   |
  |    layer3.0.bn2        |    0.512K            |    0.251M  |    0         |
  |    layer3.0.conv3      |    0.262M            |    51.38M  |    0.201M    |
  |    layer3.0.bn3        |    2.048K            |    1.004M  |    0         |
  |    layer3.0.downsample |    0.526M            |    0.104G  |    0.201M    |
  |   layer3.1             |   1.117M             |   0.22G    |   0.301M     |
  |    layer3.1.conv1      |    0.262M            |    51.38M  |    50.176K   |
  |    layer3.1.bn1        |    0.512K            |    0.251M  |    0         |
  |    layer3.1.conv2      |    0.59M             |    0.116G  |    50.176K   |
  |    layer3.1.bn2        |    0.512K            |    0.251M  |    0         |
  |    layer3.1.conv3      |    0.262M            |    51.38M  |    0.201M    |
  |    layer3.1.bn3        |    2.048K            |    1.004M  |    0         |
  |   layer3.2             |   1.117M             |   0.22G    |   0.301M     |
  |    layer3.2.conv1      |    0.262M            |    51.38M  |    50.176K   |
  |    layer3.2.bn1        |    0.512K            |    0.251M  |    0         |
  |    layer3.2.conv2      |    0.59M             |    0.116G  |    50.176K   |
  |    layer3.2.bn2        |    0.512K            |    0.251M  |    0         |
  |    layer3.2.conv3      |    0.262M            |    51.38M  |    0.201M    |
  |    layer3.2.bn3        |    2.048K            |    1.004M  |    0         |
  |   layer3.3             |   1.117M             |   0.22G    |   0.301M     |
  |    layer3.3.conv1      |    0.262M            |    51.38M  |    50.176K   |
  |    layer3.3.bn1        |    0.512K            |    0.251M  |    0         |
  |    layer3.3.conv2      |    0.59M             |    0.116G  |    50.176K   |
  |    layer3.3.bn2        |    0.512K            |    0.251M  |    0         |
  |    layer3.3.conv3      |    0.262M            |    51.38M  |    0.201M    |
  |    layer3.3.bn3        |    2.048K            |    1.004M  |    0         |
  |   layer3.4             |   1.117M             |   0.22G    |   0.301M     |
  |    layer3.4.conv1      |    0.262M            |    51.38M  |    50.176K   |
  |    layer3.4.bn1        |    0.512K            |    0.251M  |    0         |
  |    layer3.4.conv2      |    0.59M             |    0.116G  |    50.176K   |
  |    layer3.4.bn2        |    0.512K            |    0.251M  |    0         |
  |    layer3.4.conv3      |    0.262M            |    51.38M  |    0.201M    |
  |    layer3.4.bn3        |    2.048K            |    1.004M  |    0         |
  |   layer3.5             |   1.117M             |   0.22G    |   0.301M     |
  |    layer3.5.conv1      |    0.262M            |    51.38M  |    50.176K   |
  |    layer3.5.bn1        |    0.512K            |    0.251M  |    0         |
  |    layer3.5.conv2      |    0.59M             |    0.116G  |    50.176K   |
  |    layer3.5.bn2        |    0.512K            |    0.251M  |    0         |
  |    layer3.5.conv3      |    0.262M            |    51.38M  |    0.201M    |
  |    layer3.5.bn3        |    2.048K            |    1.004M  |    0         |
  |  layer4                |  14.965M             |  0.812G    |  0.627M      |
  |   layer4.0             |   6.04M              |   0.374G   |   0.326M     |
  |    layer4.0.conv1      |    0.524M            |    0.103G  |    0.1M      |
  |    layer4.0.bn1        |    1.024K            |    0.502M  |    0         |
  |    layer4.0.conv2      |    2.359M            |    0.116G  |    25.088K   |
  |    layer4.0.bn2        |    1.024K            |    0.125M  |    0         |
  |    layer4.0.conv3      |    1.049M            |    51.38M  |    0.1M      |
  |    layer4.0.bn3        |    4.096K            |    0.502M  |    0         |
  |    layer4.0.downsample |    2.101M            |    0.103G  |    0.1M      |
  |   layer4.1             |   4.463M             |   0.219G   |   0.151M     |
  |    layer4.1.conv1      |    1.049M            |    51.38M  |    25.088K   |
  |    layer4.1.bn1        |    1.024K            |    0.125M  |    0         |
  |    layer4.1.conv2      |    2.359M            |    0.116G  |    25.088K   |
  |    layer4.1.bn2        |    1.024K            |    0.125M  |    0         |
  |    layer4.1.conv3      |    1.049M            |    51.38M  |    0.1M      |
  |    layer4.1.bn3        |    4.096K            |    0.502M  |    0         |
  |   layer4.2             |   4.463M             |   0.219G   |   0.151M     |
  |    layer4.2.conv1      |    1.049M            |    51.38M  |    25.088K   |
  |    layer4.2.bn1        |    1.024K            |    0.125M  |    0         |
  |    layer4.2.conv2      |    2.359M            |    0.116G  |    25.088K   |
  |    layer4.2.bn2        |    1.024K            |    0.125M  |    0         |
  |    layer4.2.conv3      |    1.049M            |    51.38M  |    0.1M      |
  |    layer4.2.bn3        |    4.096K            |    0.502M  |    0         |
  |  fc                    |  2.049M              |  2.048M    |  1K          |
  |   fc.weight            |   (1000, 2048)       |            |              |
  |   fc.bias              |   (1000,)            |            |              |
  |  avgpool               |                      |  0.1M      |  0           |
  +------------------------+----------------------+------------+--------------+
  ```

  </details>

- 以模型结构的形式显示

  ```python
  print(analysis_results['out_arch'])
  ```

  <details>
    <summary>点击展开</summary>

  ```python
  MMResNet50(
  #params: 25.56M, #flops: 4.14G, #acts: 11.11M
  (data_preprocessor): BaseDataPreprocessor(#params: 0, #flops: N/A, #acts: N/A)
  (resnet): ResNet(
      #params: 25.56M, #flops: 4.14G, #acts: 11.11M
      (conv1): Conv2d(
      3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
      #params: 9.41K, #flops: 0.12G, #acts: 0.8M
      )
      (bn1): BatchNorm2d(
      64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
      #params: 0.13K, #flops: 4.01M, #acts: 0
      )
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
      #params: 0.22M, #flops: 0.69G, #acts: 4.42M
      (0): Bottleneck(
          #params: 75.01K, #flops: 0.24G, #acts: 2.01M
          (conv1): Conv2d(
          64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 4.1K, #flops: 12.85M, #acts: 0.2M
          )
          (bn1): BatchNorm2d(
          64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.13K, #flops: 1M, #acts: 0
          )
          (conv2): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 36.86K, #flops: 0.12G, #acts: 0.2M
          )
          (bn2): BatchNorm2d(
          64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.13K, #flops: 1M, #acts: 0
          )
          (conv3): Conv2d(
          64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 16.38K, #flops: 51.38M, #acts: 0.8M
          )
          (bn3): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 4.01M, #acts: 0
          )
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
          #params: 16.9K, #flops: 55.39M, #acts: 0.8M
          (0): Conv2d(
              64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
              #params: 16.38K, #flops: 51.38M, #acts: 0.8M
          )
          (1): BatchNorm2d(
              256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              #params: 0.51K, #flops: 4.01M, #acts: 0
          )
          )
      )
      (1): Bottleneck(
          #params: 70.4K, #flops: 0.22G, #acts: 1.2M
          (conv1): Conv2d(
          256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 16.38K, #flops: 51.38M, #acts: 0.2M
          )
          (bn1): BatchNorm2d(
          64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.13K, #flops: 1M, #acts: 0
          )
          (conv2): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 36.86K, #flops: 0.12G, #acts: 0.2M
          )
          (bn2): BatchNorm2d(
          64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.13K, #flops: 1M, #acts: 0
          )
          (conv3): Conv2d(
          64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 16.38K, #flops: 51.38M, #acts: 0.8M
          )
          (bn3): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 4.01M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
          #params: 70.4K, #flops: 0.22G, #acts: 1.2M
          (conv1): Conv2d(
          256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 16.38K, #flops: 51.38M, #acts: 0.2M
          )
          (bn1): BatchNorm2d(
          64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.13K, #flops: 1M, #acts: 0
          )
          (conv2): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 36.86K, #flops: 0.12G, #acts: 0.2M
          )
          (bn2): BatchNorm2d(
          64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.13K, #flops: 1M, #acts: 0
          )
          (conv3): Conv2d(
          64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 16.38K, #flops: 51.38M, #acts: 0.8M
          )
          (bn3): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 4.01M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      )
      (layer2): Sequential(
      #params: 1.22M, #flops: 1.04G, #acts: 3.11M
      (0): Bottleneck(
          #params: 0.38M, #flops: 0.38G, #acts: 1.3M
          (conv1): Conv2d(
          256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 32.77K, #flops: 0.1G, #acts: 0.4M
          )
          (bn1): BatchNorm2d(
          128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.26K, #flops: 2.01M, #acts: 0
          )
          (conv2): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
          #params: 0.15M, #flops: 0.12G, #acts: 0.1M
          )
          (bn2): BatchNorm2d(
          128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.26K, #flops: 0.5M, #acts: 0
          )
          (conv3): Conv2d(
          128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 65.54K, #flops: 51.38M, #acts: 0.4M
          )
          (bn3): BatchNorm2d(
          512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 1.02K, #flops: 2.01M, #acts: 0
          )
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
          #params: 0.13M, #flops: 0.1G, #acts: 0.4M
          (0): Conv2d(
              256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
              #params: 0.13M, #flops: 0.1G, #acts: 0.4M
          )
          (1): BatchNorm2d(
              512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              #params: 1.02K, #flops: 2.01M, #acts: 0
          )
          )
      )
      (1): Bottleneck(
          #params: 0.28M, #flops: 0.22G, #acts: 0.6M
          (conv1): Conv2d(
          512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 65.54K, #flops: 51.38M, #acts: 0.1M
          )
          (bn1): BatchNorm2d(
          128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.26K, #flops: 0.5M, #acts: 0
          )
          (conv2): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 0.15M, #flops: 0.12G, #acts: 0.1M
          )
          (bn2): BatchNorm2d(
          128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.26K, #flops: 0.5M, #acts: 0
          )
          (conv3): Conv2d(
          128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 65.54K, #flops: 51.38M, #acts: 0.4M
          )
          (bn3): BatchNorm2d(
          512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 1.02K, #flops: 2.01M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
          #params: 0.28M, #flops: 0.22G, #acts: 0.6M
          (conv1): Conv2d(
          512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 65.54K, #flops: 51.38M, #acts: 0.1M
          )
          (bn1): BatchNorm2d(
          128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.26K, #flops: 0.5M, #acts: 0
          )
          (conv2): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 0.15M, #flops: 0.12G, #acts: 0.1M
          )
          (bn2): BatchNorm2d(
          128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.26K, #flops: 0.5M, #acts: 0
          )
          (conv3): Conv2d(
          128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 65.54K, #flops: 51.38M, #acts: 0.4M
          )
          (bn3): BatchNorm2d(
          512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 1.02K, #flops: 2.01M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
          #params: 0.28M, #flops: 0.22G, #acts: 0.6M
          (conv1): Conv2d(
          512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 65.54K, #flops: 51.38M, #acts: 0.1M
          )
          (bn1): BatchNorm2d(
          128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.26K, #flops: 0.5M, #acts: 0
          )
          (conv2): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 0.15M, #flops: 0.12G, #acts: 0.1M
          )
          (bn2): BatchNorm2d(
          128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.26K, #flops: 0.5M, #acts: 0
          )
          (conv3): Conv2d(
          128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 65.54K, #flops: 51.38M, #acts: 0.4M
          )
          (bn3): BatchNorm2d(
          512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 1.02K, #flops: 2.01M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      )
      (layer3): Sequential(
      #params: 7.1M, #flops: 1.48G, #acts: 2.16M
      (0): Bottleneck(
          #params: 1.51M, #flops: 0.38G, #acts: 0.65M
          (conv1): Conv2d(
          512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.13M, #flops: 0.1G, #acts: 0.2M
          )
          (bn1): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 1M, #acts: 0
          )
          (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
          #params: 0.59M, #flops: 0.12G, #acts: 50.18K
          )
          (bn2): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 0.2M
          )
          (bn3): BatchNorm2d(
          1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 2.05K, #flops: 1M, #acts: 0
          )
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
          #params: 0.53M, #flops: 0.1G, #acts: 0.2M
          (0): Conv2d(
              512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False
              #params: 0.52M, #flops: 0.1G, #acts: 0.2M
          )
          (1): BatchNorm2d(
              1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              #params: 2.05K, #flops: 1M, #acts: 0
          )
          )
      )
      (1): Bottleneck(
          #params: 1.12M, #flops: 0.22G, #acts: 0.3M
          (conv1): Conv2d(
          1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 50.18K
          )
          (bn1): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 0.59M, #flops: 0.12G, #acts: 50.18K
          )
          (bn2): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 0.2M
          )
          (bn3): BatchNorm2d(
          1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 2.05K, #flops: 1M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
          #params: 1.12M, #flops: 0.22G, #acts: 0.3M
          (conv1): Conv2d(
          1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 50.18K
          )
          (bn1): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 0.59M, #flops: 0.12G, #acts: 50.18K
          )
          (bn2): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 0.2M
          )
          (bn3): BatchNorm2d(
          1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 2.05K, #flops: 1M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
          #params: 1.12M, #flops: 0.22G, #acts: 0.3M
          (conv1): Conv2d(
          1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 50.18K
          )
          (bn1): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 0.59M, #flops: 0.12G, #acts: 50.18K
          )
          (bn2): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 0.2M
          )
          (bn3): BatchNorm2d(
          1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 2.05K, #flops: 1M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      (4): Bottleneck(
          #params: 1.12M, #flops: 0.22G, #acts: 0.3M
          (conv1): Conv2d(
          1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 50.18K
          )
          (bn1): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 0.59M, #flops: 0.12G, #acts: 50.18K
          )
          (bn2): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 0.2M
          )
          (bn3): BatchNorm2d(
          1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 2.05K, #flops: 1M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      (5): Bottleneck(
          #params: 1.12M, #flops: 0.22G, #acts: 0.3M
          (conv1): Conv2d(
          1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 50.18K
          )
          (bn1): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 0.59M, #flops: 0.12G, #acts: 50.18K
          )
          (bn2): BatchNorm2d(
          256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 0.51K, #flops: 0.25M, #acts: 0
          )
          (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.26M, #flops: 51.38M, #acts: 0.2M
          )
          (bn3): BatchNorm2d(
          1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 2.05K, #flops: 1M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      )
      (layer4): Sequential(
      #params: 14.96M, #flops: 0.81G, #acts: 0.63M
      (0): Bottleneck(
          #params: 6.04M, #flops: 0.37G, #acts: 0.33M
          (conv1): Conv2d(
          1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 0.52M, #flops: 0.1G, #acts: 0.1M
          )
          (bn1): BatchNorm2d(
          512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 1.02K, #flops: 0.5M, #acts: 0
          )
          (conv2): Conv2d(
          512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
          #params: 2.36M, #flops: 0.12G, #acts: 25.09K
          )
          (bn2): BatchNorm2d(
          512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 1.02K, #flops: 0.13M, #acts: 0
          )
          (conv3): Conv2d(
          512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 1.05M, #flops: 51.38M, #acts: 0.1M
          )
          (bn3): BatchNorm2d(
          2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 4.1K, #flops: 0.5M, #acts: 0
          )
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
          #params: 2.1M, #flops: 0.1G, #acts: 0.1M
          (0): Conv2d(
              1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False
              #params: 2.1M, #flops: 0.1G, #acts: 0.1M
          )
          (1): BatchNorm2d(
              2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              #params: 4.1K, #flops: 0.5M, #acts: 0
          )
          )
      )
      (1): Bottleneck(
          #params: 4.46M, #flops: 0.22G, #acts: 0.15M
          (conv1): Conv2d(
          2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 1.05M, #flops: 51.38M, #acts: 25.09K
          )
          (bn1): BatchNorm2d(
          512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 1.02K, #flops: 0.13M, #acts: 0
          )
          (conv2): Conv2d(
          512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 2.36M, #flops: 0.12G, #acts: 25.09K
          )
          (bn2): BatchNorm2d(
          512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 1.02K, #flops: 0.13M, #acts: 0
          )
          (conv3): Conv2d(
          512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 1.05M, #flops: 51.38M, #acts: 0.1M
          )
          (bn3): BatchNorm2d(
          2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 4.1K, #flops: 0.5M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
          #params: 4.46M, #flops: 0.22G, #acts: 0.15M
          (conv1): Conv2d(
          2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 1.05M, #flops: 51.38M, #acts: 25.09K
          )
          (bn1): BatchNorm2d(
          512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 1.02K, #flops: 0.13M, #acts: 0
          )
          (conv2): Conv2d(
          512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          #params: 2.36M, #flops: 0.12G, #acts: 25.09K
          )
          (bn2): BatchNorm2d(
          512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 1.02K, #flops: 0.13M, #acts: 0
          )
          (conv3): Conv2d(
          512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
          #params: 1.05M, #flops: 51.38M, #acts: 0.1M
          )
          (bn3): BatchNorm2d(
          2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
          #params: 4.1K, #flops: 0.5M, #acts: 0
          )
          (relu): ReLU(inplace=True)
      )
      )
      (avgpool): AdaptiveAvgPool2d(
      output_size=(1, 1)
      #params: 0, #flops: 0.1M, #acts: 0
      )
      (fc): Linear(
      in_features=2048, out_features=1000, bias=True
      #params: 2.05M, #flops: 2.05M, #acts: 1K
      )
  )
  )
  ```

  </details>

- 总的计算量

  ```python
  print("Model Flops:{}".format(analysis_results['flops_str']))
  # Model Flops:4.145G
  ```

- 总的参数量

  ```python
  print("Model Parameters:{}".format(analysis_results['params_str']))
  # Model Parameters:25.557M
  ```

关于模型计算量和参数量的定义以及更多用法请阅读[模型复杂度分析](../advanced_tutorials/model_analysis.md)。
