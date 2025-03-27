# SEResNet-X

[![State-of-the-art Shitcode](https://img.shields.io/static/v1?label=State-of-the-art&message=Shitcode&color=7B5804)](https://github.com/trekhleb/state-of-the-art-shitcode)
## Intro
SEResNet-X is an image classification model based on ResNet with Squeeze-and-Excitation Blocks and SiLU.

Seems to work fine.

X means there are 10 layers and blocks in total.

Model in `/weights` is trained upon this dataset:

> **Aerial Landscape Images**
> https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset

## Specs
|Model|Image size<br>(Pixels)|Accuracy|Params|Flops|
|-|-|-|-|-|
|SEResNet-X-Standard|256x256|~ 97%<br>@epoch=150|4.96M|3.58G|

## Arch
### SEBlock
Input → Linear → SiLU → Linear → Sigmoid → Output
### ResBlock
![](images/ResBlockArch.png)

### Network
![](images/NetworkArch.png)

## Conf Mat

![Confusion Mat](images/confusion_matrix.png)