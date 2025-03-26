# SEResNet-X

## Intro
SEResNet-X is an image classification model based on ResNet with Squeeze-and-Excitation Blocks and SiLU.

Seems to work fine.

X means there are 10 layers in total.

## Specs
|Model|Image size<br>(Pixels)|Accuracy|params|
|-|-|-|-|
|SEResNet-X-Standard|256x256|~94.5%<br>@epoch=61|4.96M|

## Arch
### SEBlock
Input → Linear → SiLU → Linear → Sigmoid → Output
### ResBlock
![](images/ResBlockArch.png)

### Network
![](images/NetworkArch.png)

## Conf Mat

![Confusion Mat](images/confusion_matrix.png)