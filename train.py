from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
from PIL import ImageFile
import numpy as np
from torchvision.models import VGG16_Weights

ImageFile.LOAD_TRUNCATED_IMAGES = True

# データ前処理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# データローダ
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        # 大文字小文字を区別せずに画像を取得
        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.JPG"))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.zeros(1)  # 正常データ用のダミーラベル
    
train_dataset = CustomImageDataset("dataset/train/normal", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


# モデルの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(weights=VGG16_Weights.DEFAULT)
for param in model.features.parameters():
    param.requires_grad = False

# `classifier` をカスタマイズ
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 1024),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(1024, 32),
    nn.Sigmoid()
)

# モデルをGPUに転送
model = model.to(device)

# 損失関数とオプティマイザ

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学習プロセス
model.train()
center = torch.zeros(32).to(device)  # 超球の中心点
with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        outputs = model(images)
        center += outputs.sum(dim=0)
    center /= len(train_loader.dataset)  # 中心点の平均を計算

for epoch in range(10):  # 10エポック
    epoch_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        outputs = model(images)

        # 損失の計算
        loss = criterion(outputs, center.unsqueeze(0).expand_as(outputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

# モデルと中心点を保存
torch.save(model.state_dict(), "deep_svdd_model.pth")
torch.save(center, "center.pth")
