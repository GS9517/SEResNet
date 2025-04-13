# SEResNet10

[![State-of-the-art Shitcode](https://img.shields.io/static/v1?label=State-of-the-art&message=Shitcode&color=7B5804)](https://github.com/trekhleb/state-of-the-art-shitcode)
## Intro
SEResNet10 is an image classification model based on ResNet with Squeeze-and-Excitation Blocks and SiLU.

Seems to work fine.

There are 10 layers/blocks in total.

Model in `/weights` is trained upon this dataset:

> **Aerial Landscape Images**
> https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset

## Specs
Comparation with YOLO11-cls:
|Model|Image size<br>(Pixels)|Accuracy|Params|Flops|
|-|-|-|-|-|
|SEResNet10 |256x256|~ 97.8%<br>(@epoch=150)|4.96M|0.35B|
|YOLO11n-cls|224x224|~ 89.4%                |1.6M |0.5B |
|YOLO11s-cls|224x224|~ 92.7%                |5.5M |1.6B |

## How to Use

### Install Requirements
```shell
$ pip install -r requirements.txt
```

### Train
```shell
$ python3 train.py [-a | --augmentatioin] [-u | --unbalanced-dataset] [-h | --help]
```
`[-a | --augmentation]`: enable data augmentation

`[-u | --unbalanced-dataset]`: enable unbalanced dataset

`[-h | --help]`: print help menu

A running outcome will be generated under `./run/trainXX`

### Test
```shell
$ python3 test.py
```
A running outcome will be generated under `./run/testXX`

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

[1]	Elfwing, S., Uchibe, E. & Doya, K. 2018, 'Sigmoid-weighted linear units for neural network function approximation in reinforcement learning', Neurocomputing, vol. 298, pp. 166-174, doi: 10.1016/j.neucom.2018.01.063.

[2]	Helmholtz, H. 1867, Handbuch der physiologischen Optik, Voss.

[3]	He, K., Zhang, X., Ren, S. and Sun, J. (2015) 'Deep Residual Learning for Image Recognition'. arXiv preprint arXiv:1512.03385. Available at: https://arxiv.org/abs/1512.03385 [Accessed 27 Mar. 2025]. 

[4]	Hu, J., Shen, L., Albanie, S., Sun, G. & Wu, E. 2018, 'Squeeze-and-Excitation Networks', IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 8, pp. 2011-2023, doi: 10.1109/TPAMI.2019.2913372.

[5]	LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. 1998, 'Gradient-based learning applied to document recognition', Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, doi: 10.1109/5.726791.

[6]	Liu, Y., Cheng, M. and Lapata, M. (2017) Learning Structured Text Representations. arXiv preprint arXiv:1709.01507v4. Available at: https://arxiv.org/abs/1709.01507v4 [Accessed 27 Mar. 2025].

[7]	Marr, D. 1982, Vision: A computational investigation into the human representation and processing of visual information, W.H. Freeman and Company.

[8]	Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and Polosukhin, I. (2017) 'Attention is All You Need'. arXiv preprint arXiv:1706.03762. Available at: https://arxiv.org/abs/1706.03762 [Accessed 27 Mar. 2025].

[9]	Zhang, D., Liu, H., Yang, Y., Wang, H., Liang, Y., Ye, Z., Qin, Z., Huang, T.S. and Zhuang, Y. (2018) 'Neural machine translation with deep attention'. Neural Networks, 100, pp. 24-35. doi: 10.1016/j.neunet.2017.12.010.
