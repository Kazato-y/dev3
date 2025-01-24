import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DeepSVDDNet(nn.Module):
    """
    Deep SVDD Network with convolutional layers for feature extraction
    and fully connected layers for dimensionality reduction.
    """

    def __init__(self, input_channels=3, output_dim=32, hidden_activation=None, output_activation=None):
        super().__init__()
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            self.hidden_activation,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.hidden_activation,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            self.hidden_activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, output_dim),
            self.output_activation,
        )

    def forward(self, inputs):
        conv_out = self.conv(inputs)
        flattened = conv_out.view(conv_out.size(0), -1)
        embed = self.fc(flattened)
        return embed


class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from directories.
    Assumes the following structure:
    root_dir/
        normal/
        anomaly/
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load normal data
        normal_dir = os.path.join(root_dir, "normal")
        if os.path.exists(normal_dir):
            for file_name in os.listdir(normal_dir):
                if file_name.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                    self.image_paths.append(os.path.join(normal_dir, file_name))
                    self.labels.append(0)

        # Load anomaly data (optional)
        anomaly_dir = os.path.join(root_dir, "anomaly")
        if os.path.exists(anomaly_dir):
            for file_name in os.listdir(anomaly_dir):
                if file_name.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                    self.image_paths.append(os.path.join(anomaly_dir, file_name))
                    self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label