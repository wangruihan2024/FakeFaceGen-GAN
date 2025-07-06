- 解释实现过程

以$64*64$分辨率为例:
在`preprocess.py`中先预处理人脸图片和数据集，裁剪缩放到64*64, 经过dataloader生成训练批次。 在`network.py`中定义生成器和输入器两个神经网络，定义多层反卷积和卷积的环境。 在`train.py`中先设置参数然后训练epoch次，保存最后的结果。

在`network.py`中discriminator输入一个$3*64*64$的图最后输出一个概率0-1表示是真的图片的概率；generator是一个反卷积的神经网络，把一个随机噪声向量变成一张假人脸图片，经过多层反卷积将图片逐渐放大，最后输出$3*64*64$的图片。

训练目的在于让generator产生出的图discriminator判断不出来是否是生成的，即$G(D(x)) = 0.5$

- 解释 `train.py` 的大致思想

```python
img_dim = 64
lr = 0.0002
epochs = 5
batch_size = 128
G_DIMENSION = 100
beta1 = 0.5
beta2 = 0.999
output_path = 'output'
real_label = 1
fake_label = 0
```
设置图像分辨率$img\_dim$，
学习率$lr$, 
要训练的次数$epoch$, 
每批的训练数据量$batch\_size$, Adam优化器的参数$beta1, beta2$等
```python
netD = Discriminator().to(device)
netG = Generator().to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
```
定义两个模型， 定义$Loss$函数，并且用Adam优化器使训练结果收敛

接下来在每一个$epoch$中，分为训练$discriminatorD$，$generatorG$

- Discriminator D
    - 用真实图像训练，目标标签为1
    - 用生成图片训练，目标标签为0
    - 计算损失，反向传播，优化参数

- Generator G
    - 生成假图片训练，目标标签为1
    - 计算损失，反向传播

- 每隔一定步数计算保存一次损失曲线

最后训练结束完成画Generator和Discriminator的loss曲线