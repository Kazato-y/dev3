import torch
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
import torch.nn as nn
from pathlib import Path
from PIL import ImageFile, Image
from tqdm import tqdm  # 進捗バー用

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
        label = "anomaly" if "anomaly" in image_path.parts else "normal"  # パス名でラベルを判断
        if self.transform:
            image = self.transform(image)
        return image, label, image_path.name  # ラベルとファイル名を返す

# モデルのロード
# モデルのロード
def load_model(weights_path="weights.pth", center_path="center.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
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

    # `weights_only=True` を明示的に指定してロード
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model = model.to(device)
    model.eval()

    center = torch.load(center_path, weights_only=True).to(device)

    return model, center, device

# 異常検知関数
def detect_anomaly(model, center, dataloader, device, dataset_type="", threshold=0.1):
    print(f"\nDetecting anomalies in {dataset_type} dataset:")
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Processing {dataset_type}", leave=False)
        for images, labels, image_names in progress_bar:
            images = images.to(device)  # 修正: 明示的に device を使用
            outputs = model(images)

            # 超球中心との距離を計算
            distances = torch.norm(outputs - center.unsqueeze(0), dim=1)
            for image_name, label, distance in zip(image_names, labels, distances):
                result = "Anomalous" if distance > threshold else "Normal"
                print(f"{image_name} ({label}): {result} (Distance: {distance:.4f})")

# `test.py` を直接実行した場合のみ動作する処理
def main():
    test_normal_dataset = CustomImageDataset("dataset/test/normal", transform=transform)
    test_anomaly_dataset = CustomImageDataset("dataset/test/anomaly", transform=transform)

    test_normal_loader = torch.utils.data.DataLoader(test_normal_dataset, batch_size=4, shuffle=False)
    test_anomaly_loader = torch.utils.data.DataLoader(test_anomaly_dataset, batch_size=4, shuffle=False)

    model, center, device = load_model()

    detect_anomaly(model, center, test_normal_loader, device, "normal")
    detect_anomaly(model, center, test_anomaly_loader, device, "anomaly")

if __name__ == "__main__":
    main()