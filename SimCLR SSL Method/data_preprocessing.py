import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SimCLR Augmentations (Stronger)
class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),  # More aggressive cropping
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),  # Less frequent vertical flips
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.5),
            transforms.RandomApply([transforms.RandomSolarize(threshold=128)], p=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __call__(self, img):
        img1, img2 = self.transform(img), self.transform(img)
        return img1.to(device), img2.to(device)  # Move to GPU

# Custom Dataset for Unlabeled Data
class SimCLRDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = SimCLRTransform()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        img1, img2 = self.transform(img)
        return img1, img2  # Already on GPU

# Load Data
dataset_path = r"E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\Task 2 - SimCLR Implementation\BreaKHis_unlabeled"
dataset = SimCLRDataset(dataset_path)

dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=0)  # num_workers=0 for Windows

# Test the DataLoader
for img1, img2 in dataloader:
    print(f"Batch size: {img1.shape}, {img2.shape}")
    break  # Just check one batch
