import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from svdd import DeepSVDDNet, CustomImageDataset

# データ前処理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# データローダ
train_dataset = CustomImageDataset("dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# モデルの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepSVDDNet(
    input_channels=3,
    output_dim=32,
    hidden_activation=torch.nn.ReLU(),
    output_activation=torch.nn.Sigmoid()
).to(device)

# 損失関数とオプティマイザ
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学習プロセス
model.train()
for epoch in range(10):  # 10エポック
    epoch_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        outputs = model(images)

        # 損失の計算
        loss = criterion(outputs, torch.zeros_like(outputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

# モデルを保存
torch.save(model.state_dict(), "deep_svdd_model.pth")