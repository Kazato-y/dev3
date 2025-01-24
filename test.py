import torch
import numpy as np
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
test_dataset = CustomImageDataset("dataset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# モデルのロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepSVDDNet(
    input_channels=3,
    output_dim=32,
    hidden_activation=torch.nn.ReLU(),
    output_activation=torch.nn.Sigmoid()
).to(device)
model.load_state_dict(torch.load("deep_svdd_model.pth"))
model.eval()

# 中心点の計算（学習データから事前に取得済みと仮定）
center = torch.zeros(32).to(device)

# 異常スコアの計算
scores = []
labels = []
with torch.no_grad():
    for images, label in test_loader:
        images = images.to(device)
        outputs = model(images)
        distances = torch.sum((outputs - center) ** 2, dim=1)
        scores.extend(distances.cpu().numpy())
        labels.extend(label.numpy())

# 結果表示
threshold = np.percentile(scores, 95)  # 上位5%を異常と判定
print(f"Threshold: {threshold:.4f}")

for i, (score, label) in enumerate(zip(scores, labels)):
    pred = 1 if score > threshold else 0
    print(f"Sample {i+1}: Anomaly Score = {score:.4f}, Predicted = {pred}, Label = {label}")