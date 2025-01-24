import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from svdd import DeepSVDDNet, CustomImageDataset

# データ前処理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# データセットとデータローダの設定
train_dataset = CustomImageDataset("dataset/train", transform=transform)
test_dataset = CustomImageDataset("dataset/test", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 中心点の計算（仮にゼロベクトルとして定義）
center = torch.zeros(32).to(device)

# 設定するエポック数のリスト
epochs_list = [1, 2, 3]
results = []

# 各エポック数での学習とテスト
for num_epochs in epochs_list:
    print(f"Training with {num_epochs} epochs...")
    # モデルの初期化
    model = DeepSVDDNet(
        input_channels=3,
        output_dim=32,
        hidden_activation=torch.nn.ReLU(),
        output_activation=torch.nn.Sigmoid()
    ).to(device)

    # 損失関数とオプティマイザ
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 学習ループ
    train_losses = []
    model.train()
    for epoch in range(num_epochs):
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
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # モデルの評価
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for images, label in test_loader:
            images = images.to(device)
            outputs = model(images)
            distances = torch.sum((outputs - center) ** 2, dim=1)
            scores.extend(distances.cpu().numpy())
            labels.extend(label.numpy())

    # 閾値と異常検出精度の計算
    threshold = np.percentile(scores, 95)
    predictions = [1 if score > threshold else 0 for score in scores]
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    results.append((num_epochs, train_losses, accuracy))

    print(f"Threshold: {threshold:.4f}, Accuracy: {accuracy:.4f}")

# 可視化
plt.figure(figsize=(12, 6))

# 学習損失のプロット
for num_epochs, train_losses, _ in results:
    plt.plot(range(1, len(train_losses) + 1), train_losses, label=f'{num_epochs} epochs')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# 異常検出精度のプロット
accuracies = [result[2] for result in results]
plt.figure(figsize=(8, 4))
plt.bar([f"{result[0]} epochs" for result in results], accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Anomaly Detection Accuracy')
plt.show()