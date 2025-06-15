import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display
from network import *
from dataloader import *  # 假设您已经定义了 train_loader

# 超参数
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

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型
netD = Discriminator().to(device)
netG = Generator().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

# 训练过程
losses = [[], []]
# plt.ion()
plt.ioff()
now = 0
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch_id, data in enumerate(train_loader):
        ############################
        # (1) 更新判别器 D
        ###########################
        optimizerD.zero_grad()
        # real_cpu = data[0].to(device).unsqueeze(0)
        real_cpu = data.to(device)
        # print(f"real_cpu.shape{real_cpu.shape}")
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        label = label.squeeze()
        # print(f"label.shape{label.shape}")
        output = netD(real_cpu)
        output = output.view(-1)
        # print(f"output.shape{output.shape}")
        
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, G_DIMENSION, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) 更新生成器 G
        ###########################
        optimizerG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # 保存损失
        losses[0].append(errD.item())
        losses[1].append(errG.item())

        # 每隔一定批次保存生成的图像
        if batch_id % 100 == 0:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with torch.no_grad():
                fake = netG(noise).detach().cpu()
            imgs = []
            plt.figure(figsize=(15, 15))
            try:
                for i in range(100):
                    image = fake[i].numpy().transpose(1, 2, 0)
                    image = np.where(image > 0, image, 0)
                    plt.subplot(10, 10, i + 1)
                    plt.imshow(image, vmin=-1, vmax=1)
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplots_adjust(wspace=0.1, hspace=0.1)
                msg = 'Epoch ID={0} Batch ID={1} \n\n D-Loss={2} G-Loss={3}'.format(epoch, batch_id, errD.item(), errG.item())
                plt.suptitle(msg, fontsize=20)
                # plt.draw()
                plt.savefig('{}/{:04d}_{:04d}.png'.format(output_path, epoch, batch_id), bbox_inches='tight')
                # plt.pause(5)
                plt.close()
                display.clear_output(wait=True)
            except IOError:
                print(IOError)

plt.close()
plt.figure(figsize=(15, 6))
x = np.arange(len(losses[0]))
plt.title('Generator and Discriminator Loss During Training')
plt.xlabel('Number of Batch')
plt.plot(x, np.array(losses[0]), label='D Loss')
plt.plot(x, np.array(losses[1]), label='G Loss')
plt.legend()
plt.savefig('Generator and Discriminator Loss During Training.png')
plt.show()
torch.save(netG.state_dict(), "generator.params")
plt.close()