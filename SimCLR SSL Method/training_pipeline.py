import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True

def get_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        img1, img2 = self.transform(img), self.transform(img)
        return img1.to(device, non_blocking=True), img2.to(device, non_blocking=True)

class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return projections

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T)
    labels = torch.arange(2 * batch_size, dtype=torch.long, device=device)
    loss_fn = nn.CrossEntropyLoss()
    logits = sim_matrix / temperature
    loss = loss_fn(logits, labels)
    return loss

def train(model, dataloader, optimizer, num_epochs=10):
    best_loss = float('inf')
    best_model_path = "best_model.pth"
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            optimizer.zero_grad()
            z1, z2 = model(x1), model(x2)
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {avg_loss:.4f}")
        
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")

# Data and Training Setup
image_dir = r"E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\Task 2 - SimCLR Implementation\BreaKHis_unlabeled"
transform = get_transforms()
dataset = ImageDataset(image_dir, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

model = SimCLR().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, dataloader, optimizer, num_epochs=10)
