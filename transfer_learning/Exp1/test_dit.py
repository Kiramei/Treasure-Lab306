import os
import time

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import cv2

T = 300  # 加噪最大步数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备


# 手写数字
class FS2K(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.origin = os.listdir('dataset/FS2K/train/origin')
        self.target = os.listdir('dataset/FS2K/train/target')

    def __len__(self):
        return len(self.origin)

    def __getitem__(self, index):
        origin, target = self.origin[index], self.target[index]
        origin = cv2.imread('dataset/FS2K/train/origin/' + origin, cv2.IMREAD_GRAYSCALE)
        origin = torch.Tensor(origin).unsqueeze(2).to(DEVICE) / 255.0
        target = cv2.imread('dataset/FS2K/train/target/' + target, cv2.IMREAD_GRAYSCALE)
        target = torch.Tensor(target).unsqueeze(2).to(DEVICE) / 255.0
        return origin, target


class TimeEmbedding(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.half_emb_size = emb_size // 2
        half_emb = torch.exp(torch.arange(self.half_emb_size) * (-1 * math.log(10000) / (self.half_emb_size - 1)))
        self.register_buffer('half_emb', half_emb)

    def forward(self, t):
        t = t.view(t.size(0), 1)
        half_emb = self.half_emb.unsqueeze(0).expand(t.size(0), self.half_emb_size)
        half_emb_t = half_emb * t
        embs_t = torch.cat((half_emb_t.sin(), half_emb_t.cos()), dim=-1)
        return embs_t


# 前向diffusion计算参数
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# betas = torch.linspace(0.0001, 0.02, T).to(DEVICE)  # (T,)
betas = cosine_beta_schedule(T).to(DEVICE)  # (T,)
alphas = 1 - betas  # (T,)
alphas_cumprod = torch.cumprod(alphas, dim=-1)  # alpha_t累乘 (T,)    [a1,a2,a3,....] ->  [a1,a1*a2,a1*a2*a3,.....]
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]).to(DEVICE), alphas_cumprod[:-1]),
                                dim=-1)  # alpha_t-1累乘 (T,),  [1,a1,a1*a2,a1*a2*a3,.....]
variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)  # denoise用的方差   (T,)


# 执行前向加噪
def forward_add_noise(x, t):  # batch_x: (batch,channel,height,width), batch_t: (batch_size,)
    noise = torch.randn_like(x)  # 为每张图片生成第t步的高斯噪音   (batch,channel,height,width)
    batch_alphas_cumprod = alphas_cumprod[t].view(x.size(0), 1, 1, 1)
    x = torch.sqrt(batch_alphas_cumprod) * x + torch.sqrt(1 - batch_alphas_cumprod) * noise  # 基于公式直接生成第t步加噪后图片
    return x, noise


from torch import nn
import torch
import math


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class DiTBlock(nn.Module):
    def __init__(self, emb_size, nhead):
        super().__init__()

        self.emb_size = emb_size
        self.nhead = nhead

        # conditioning
        self.gamma1 = nn.Linear(emb_size, emb_size)
        self.beta1 = nn.Linear(emb_size, emb_size)
        self.alpha1 = nn.Linear(emb_size, emb_size)
        self.gamma2 = nn.Linear(emb_size, emb_size)
        self.beta2 = nn.Linear(emb_size, emb_size)
        self.alpha2 = nn.Linear(emb_size, emb_size)

        # layer norm
        self.ln1 = DyT(emb_size)
        self.ln2 = DyT(emb_size)
        # self.ln1 = nn.LayerNorm(emb_size)
        # self.ln2 = nn.LayerNorm(emb_size)

        # multi-head self-attention
        self.wq = nn.Linear(emb_size, nhead * emb_size)  # (batch,seq_len,nhead*emb_size)
        self.wk = nn.Linear(emb_size, nhead * emb_size)  # (batch,seq_len,nhead*emb_size)
        self.wv = nn.Linear(emb_size, nhead * emb_size)  # (batch,seq_len,nhead*emb_size)
        self.lv = nn.Linear(nhead * emb_size, emb_size)

        # feed-forward
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )

    def forward(self, x, cond):  # x:(batch,seq_len,emb_size), cond:(batch,emb_size)
        # conditioning (batch,emb_size)
        gamma1_val = self.gamma1(cond)
        beta1_val = self.beta1(cond)
        alpha1_val = self.alpha1(cond)
        gamma2_val = self.gamma2(cond)
        beta2_val = self.beta2(cond)
        alpha2_val = self.alpha2(cond)

        # layer norm
        y = self.ln1(x)  # (batch,seq_len,emb_size)

        # scale&shift
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1)

        # attention
        q = self.wq(y)  # (batch,seq_len,nhead*emb_size)
        k = self.wk(y)  # (batch,seq_len,nhead*emb_size)
        v = self.wv(y)  # (batch,seq_len,nhead*emb_size)
        q = q.view(q.size(0), q.size(1), self.nhead, self.emb_size).permute(0, 2, 1,
                                                                            3)  # (batch,nhead,seq_len,emb_size)
        k = k.view(k.size(0), k.size(1), self.nhead, self.emb_size).permute(0, 2, 3,
                                                                            1)  # (batch,nhead,emb_size,seq_len)
        v = v.view(v.size(0), v.size(1), self.nhead, self.emb_size).permute(0, 2, 1,
                                                                            3)  # (batch,nhead,seq_len,emb_size)
        attn = q @ k / math.sqrt(q.size(2))  # (batch,nhead,seq_len,seq_len)
        attn = torch.softmax(attn, dim=-1)  # (batch,nhead,seq_len,seq_len)
        y = attn @ v  # (batch,nhead,seq_len,emb_size)
        y = y.permute(0, 2, 1, 3)  # (batch,seq_len,nhead,emb_size)
        y = y.reshape(y.size(0), y.size(1), y.size(2) * y.size(3))  # (batch,seq_len,nhead*emb_size)
        y = self.lv(y)  # (batch,seq_len,emb_size)

        # scale
        y = y * alpha1_val.unsqueeze(1)
        # redisual
        y = x + y

        # layer norm
        z = self.ln2(y)
        # scale&shift
        z = z * (1 + gamma2_val.unsqueeze(1)) + beta2_val.unsqueeze(1)
        # feef-forward
        z = self.ff(z)
        # scale
        z = z * alpha2_val.unsqueeze(1)
        # residual
        return y + z


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备


def backward_denoise(model, x, y):
    steps = [x.clone(), ]

    global alphas, alphas_cumprod, variance

    x = x.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    variance = variance.to(DEVICE)
    y = y.to(DEVICE)

    model.eval()
    with torch.no_grad():
        for time in range(T - 1, -1, -1):
            t = torch.full((x.size(0),), time).to(DEVICE)

            # 预测x_t时刻的噪音
            noise = model(x, t, y)

            # 生成t-1时刻的图像
            shape = (x.size(0), 1, 1, 1)
            mean = 1 / torch.sqrt(alphas[t].view(*shape)) * \
                   (
                           x -
                           (1 - alphas[t].view(*shape)) / torch.sqrt(1 - alphas_cumprod[t].view(*shape)) * noise
                   )
            if time != 0:
                x = mean + \
                    torch.randn_like(x) * \
                    torch.sqrt(variance[t].view(*shape))
            else:
                x = mean
            x = torch.clamp(x, -1.0, 1.0).detach()
            steps.append(x)
    return steps


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class DiT(nn.Module):
    def __init__(self, img_size, patch_size, channel, emb_size, label_num, dit_num, head):
        super().__init__()

        self.patch_size = patch_size
        self.patch_count = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.channel = channel

        # patchify
        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel * patch_size ** 2, kernel_size=patch_size,
                              padding=0, stride=patch_size)
        self.patch_emb = nn.Linear(in_features=channel * patch_size ** 2, out_features=emb_size)
        self.patch_pos_emb = nn.Parameter(torch.rand(1, self.patch_count[0] * self.patch_count[1], emb_size))

        # time emb
        self.time_emb = nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        # label emb
        self.label_emb = nn.Embedding(num_embeddings=label_num, embedding_dim=emb_size)

        # DiT Blocks
        self.dits = nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size, head))

        # layer norm
        self.ln = DyT(emb_size)
        # self.ln = nn.LayerNorm(emb_size)

        # linear back to patch
        self.linear = nn.Linear(emb_size, channel * patch_size ** 2)

        self.cond_conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 64, 1, 1)
            nn.Flatten(),  # (B, 64)
            nn.Linear(64, emb_size)  # 映射到 emb_size 大小
        )

    def forward(self, x, t, img_cond):  # x:(batch,channel,height,width)   t:(batch,)  y:(batch,)
        # label emb
        cond_emb = self.cond_conv(img_cond)  # (batch,emb_size)
        # time emb
        t_emb = self.time_emb(t)  # (batch,emb_size)

        # condition emb
        cond = cond_emb + t_emb
        cond = t_emb

        # patch emb
        x = self.conv(x)  # (batch,new_channel,patch_count,patch_count)
        x = x.permute(0, 2, 3, 1)  # (batch,patch_count,patch_count,new_channel)
        x = x.view(x.size(0), self.patch_count[0] * self.patch_count[1],
                   x.size(3))  # (batch,patch_count**2,new_channel)

        x = self.patch_emb(x)  # (batch,patch_count**2,emb_size)
        x = x + self.patch_pos_emb  # (batch,patch_count**2,emb_size)

        # dit blocks
        for dit in self.dits:
            x = dit(x, cond)

        # # layer norm
        x = self.ln(x)  # (batch,patch_count**2,emb_size)

        # # linear back to patch
        x = self.linear(x)  # (batch,patch_count**2,channel*patch_size*patch_size)

        # reshape
        x = x.view(x.size(0), self.patch_count[0], self.patch_count[1], self.channel, self.patch_size,
                   self.patch_size)  # (batch,patch_count,patch_count,channel,patch_size,patch_size)
        x = x.permute(0, 3, 1, 2, 4, 5)  # (batch,channel,patch_count(H),patch_count(W),patch_size(H),patch_size(W))
        x = x.permute(0, 1, 2, 4, 3, 5)  # (batch,channel,patch_count(H),patch_size(H),patch_count(W),patch_size(W))
        x = x.reshape(x.size(0), self.channel, self.patch_count[0] * self.patch_size,
                      self.patch_count[1] * self.patch_size)  # (batch,channel,img_size,img_size)
        return x


# if __name__ == '__main__':
#     dit = DiT(img_size=(250, 200), patch_size=10, channel=3, emb_size=64, label_num=10, dit_num=3, head=4)
#     x = torch.rand(5, 3, 250, 200)
#     t = torch.randint(0, T, (5,))
#     y = torch.rand(5, 3, 250, 200)
#     outputs = dit(x, t, y)
#     print(outputs.shape)

# TO TRAIN
def train():
    dataset = FS2K()  # 数据集

    model = DiT(img_size=(224, 224), patch_size=14, channel=1, emb_size=128, label_num=10, dit_num=6, head=4).to(
        DEVICE)  # 模型

    try:  # 加载模型
        model.load_state_dict(torch.load('model.pth'))
    except:
        pass

    optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器

    loss_fn = nn.L1Loss()  # 损失函数(绝对值误差均值)

    '''
        训练模型
    '''

    EPOCH = 5000
    BATCH_SIZE = 128

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # 数据加载器

    model.train()

    iter_count = 0
    for epoch in range(EPOCH):
        st = time.time()
        for origin, target in dataloader:

            target = target.permute(0, 3, 1, 2)  # (batch,channel,height,width) -> (batch,height,width,channel)
            origin = origin.permute(0, 3, 1, 2)  # (batch,channel,height,width) -> (batch,height,width,channel)

            origin = origin * 2 - 1  # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应
            target = target * 2 - 1  # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应

            t = torch.randint(0, T, (origin.size(0),))  # 为每张图片生成随机t时刻

            target, noise = forward_add_noise(target, t)  # x:加噪图 noise:噪音
            pred_noise = model(target, t.to(DEVICE), origin)

            loss = loss_fn(pred_noise, noise.to(DEVICE))

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            if iter_count % 1000 == 0:
                print('epoch:{} iter:{},loss:{}'.format(epoch, iter_count, loss))
                torch.save(model.state_dict(), '.model.pth')
                os.replace('.model.pth', 'model.pth')
            iter_count += 1

        ed = time.time()
        print('epoch:{} iter:{},loss:{},time:{}'.format(epoch, iter_count, loss, ed - st))


def test():
    model = DiT(img_size=(224, 224), patch_size=14, channel=1, emb_size=256, label_num=10, dit_num=6, head=4).to(
        DEVICE)  # 模型
    try:  # 加载模型
        model.load_state_dict(torch.load('model.pth'))
    except:
        print("模型加载失败")
        return
    x = torch.rand(1, 1, 224, 224).to(DEVICE)
    image = cv2.imread(r'F:\WorkSpace\py\Experiment\tl\Exp1\dataset\FS2K\train\target\00002.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = torch.from_numpy(image).to(DEVICE) / 255.0
    image = image.unsqueeze(2)  # (height,width) -> (height,width,channel)
    image = image.permute(2, 0, 1)  # (height,width,channel) -> (channel,height,width)
    image = image.unsqueeze(0)  # (channel,height,width) -> (1,channel,height,width)
    image = image * 2 - 1  # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应

    y = image.to(DEVICE)
    steps = backward_denoise(model, x, y)
    # 绘制数量
    num_imgs = 5
    # 绘制还原过程
    plt.figure()
    for b in range(1):
        for i in range(0, num_imgs):
            idx = int(T / num_imgs) * (i + 1)
            # 像素值还原到[0,1]
            final_img = (steps[idx][b].to('cpu') + 1) / 2
            # tensor转回PIL图
            # final_img = final_img.permute(1, 2, 0)
            final_img = final_img.squeeze().numpy()
            plt.subplot(1, num_imgs, b * num_imgs + i + 1)
            final_img = (final_img * 255).astype('uint8')
            plt.imshow(final_img)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
    plt.show()


def test_noising():
    import matplotlib.pyplot as plt

    dataset = FS2K()

    x = torch.stack((dataset[0][0], dataset[1][0]), dim=0)  # 2个图片拼batch, (2,1,48,48)
    x = x.cpu()  # 转到cpu
    # 原图
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(x[0])
    plt.subplot(1, 2, 2)
    plt.imshow(x[1])
    plt.show()

    # 随机时间步
    t = torch.randint(0, T, size=(x.size(0),))
    print('t:', t)

    x = x.to(DEVICE)

    # 加噪
    x = x * 2 - 1  # [0,1]像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配
    x, noise = forward_add_noise(x, t)
    print('x:', x.size())
    print('noise:', noise.size())
    x = x.cpu()  # 转到cpu

    # 加噪图
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(((x[0] + 1) / 2))
    plt.subplot(1, 2, 2)
    plt.imshow(((x[1] + 1) / 2))
    plt.show()


if __name__ == '__main__':
    # train()
    test()
    # test_noising()
