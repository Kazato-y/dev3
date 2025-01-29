import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
import torch.nn as nn
from PIL import ImageFile, Image
from pathlib import Path

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
        return image, torch.zeros(1)  # ダミーラベル

train_dataset = CustomImageDataset("dataset/train/normal", transform=transform)

# データ分割（80%: トレーニング、20%: 検証）
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# CUDAの使用可否を確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# 写真の枚数を出力
print(f"Number of training images: {len(train_dataset)}")

# モデルの初期化
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
model = model.to(device)

# 損失関数とオプティマイザ
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学習率スケジューラ

# 超球の中心点計算
center = torch.zeros(32).to(device)
with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        outputs = model(images)
        center += outputs.sum(dim=0)
    center /= len(train_loader.dataset)  # 中心点の平均

# 早期停止の設定
best_loss = float('inf')
patience = 5
no_improve_epochs = 0

# 学習プロセス
for epoch in range(50):  # 最大 50 エポック
    model.train()
    train_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        outputs = model(images)

        # 損失の計算
        loss = criterion(outputs, center.unsqueeze(0).expand_as(outputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # 検証プロセス
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, center.unsqueeze(0).expand_as(outputs))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # 早期停止のチェック
    if val_loss < best_loss:
        best_loss = val_loss

        # **モデルごと保存**
        torch.save(model, "best_model.pth")

        # **重みだけ保存**
        torch.save(model.state_dict(), "weights.pth")

        # **超球の中心点保存**
        torch.save(center, "center.pth")

        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping!")
            break