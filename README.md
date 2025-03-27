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

## References

1. He, K., Zhang, X., Ren, S. and Sun, J. (2015) 'Deep Residual Learning for Image Recognition'. arXiv preprint arXiv:1512.03385. Available at: https://arxiv.org/abs/1512.03385 [Accessed 27 Mar. 2025].
<br>
2. Liu, Y., Cheng, M. and Lapata, M. (2017) *Learning Structured Text Representations*. arXiv preprint arXiv:1709.01507v4. Available at: https://arxiv.org/abs/1709.01507v4 [Accessed 27 Mar. 2025].
<br>
3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and Polosukhin, I. (2017) 'Attention is All You Need'. arXiv preprint arXiv:1706.03762. Available at: https://arxiv.org/abs/1706.03762 [Accessed 27 Mar. 2025].
<br>
4. Zhang, D., Liu, H., Yang, Y., Wang, H., Liang, Y., Ye, Z., Qin, Z., Huang, T.S. and Zhuang, Y. (2018) 'Neural machine translation with deep attention'. Neural Networks, 100, pp. 24-35. doi: 10.1016/j.neunet.2017.12.010.