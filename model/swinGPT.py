#%% this is from monica
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from swin_transformer import SwinTransformer


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class SwinUpsample(nn.Module):
    def __init__(self, num_classes=1):
        super(SwinUpsample, self).__init__()
        self.transformer = SwinTransformer(embed_dim=96,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=7,
                                            downscaling_factors=[4, 2, 2, 2],
                                            relative_pos_embedding=True)
        self.upsample1 = UpsamplingBlock(96, 64)
        self.upsample2 = UpsamplingBlock(64, 32)
        self.upsample3 = UpsamplingBlock(32, 16)
        self.conv = nn.Conv2d(16, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.transformer(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x

#%%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, origin_dir, label_dir, transform=None):
        self.origin_dir = origin_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_list = os.listdir(origin_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        origin_path = os.path.join(self.origin_dir, self.image_list[idx])
        label_path = os.path.join(self.label_dir, self.image_list[idx])

        origin_image = Image.open(origin_path).convert('RGB')
        label_image = Image.open(label_path).convert('RGB')

        if self.transform:
            origin_image = self.transform(origin_image)
            label_image = self.transform(label_image)

        return origin_image, label_image

# Replace with your own directories
origin_dir = 'origin'
label_dir = 'label'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = CustomDataset(origin_dir, label_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Define your Swin Transformer model here
class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        # Add your Swin Transformer layers and configurations here

    def forward(self, x):
        # Implement the forward pass for your Swin Transformer model
        return x

model = SwinTransformer()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (origin_images, label_images) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(origin_images)
        loss = criterion(outputs, label_images)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')



#%%  this is from chatgpt https://chat.openai.com/
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

from PIL import Image
import os

# 定义Swin Transformer模型
class SwinTransformer(nn.Module):
    def __init__(self, input_resolution=224, num_classes=1000):
        super(SwinTransformer, self).__init__()

        # TODO: 在这里定义Swin Transformer模型的网络结构
        # ...

    def forward(self, x):
        # TODO: 在这里实现Swin Transformer的前向传播
        # ...
        return x

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.image_dir = os.path.join(root_dir, 'origin')
        self.label_dir = os.path.join(root_dir, 'label')

        self.image_filenames = os.listdir(self.image_dir)
        self.label_filenames = os.listdir(self.label_dir)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # 添加其他必要的预处理步骤
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        image = self.transform(image)
        label = self.transform(label)

        return image, label

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # 计算损失函数
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

def main():
    # 设置训练参数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = SwinTransformer()
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建数据加载器
    dataset = CustomDataset('data')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建日志记录器
    writer = SummaryWriter('tb')

    # 训练模型
    for epoch in range(num_epochs):
        train_model(model, train_loader, criterion, optimizer, device)

        # 在TensorBoard中记录损失函数
        writer.add_scalar('Loss/train', loss.item(), epoch)

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()


