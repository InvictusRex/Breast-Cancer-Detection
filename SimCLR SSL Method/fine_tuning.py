import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset class
class ClassificationDataset(Dataset):
    def __init__(self, image_dir, transform, label_map):
        self.image_files = []
        self.labels = []
        self.transform = transform

        for label, folder in label_map.items():
            folder_path = os.path.join(image_dir, folder)
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
            self.image_files.extend(files)
            self.labels.extend([label] * len(files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)  # Add extra dimension
        return img, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define dataset paths
data_dir = "E:/Work Files/Semester Notes/Semester 4/Artificial Intelligence/BreaKHis_v1/structured"
label_map = {0: "benign", 1: "malignant"}  

# Load dataset
train_dataset = ClassificationDataset(os.path.join(data_dir, "train"), transform, label_map)
val_dataset = ClassificationDataset(os.path.join(data_dir, "val"), transform, label_map)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define SimCLR Model with classifier
class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = models.resnet18(pretrained=False)  
        self.encoder.fc = nn.Identity()  # Remove final FC layer
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Binary classification output
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

# Initialize model
model = SimCLR().to(device)

# Load pretrained encoder weights
pretrained_weights = torch.load("best_model.pth", map_location=device)
model.encoder.load_state_dict(pretrained_weights, strict=False)  # Load encoder weights only

# Freeze encoder except last ResNet block
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.encoder.layer4.parameters():  # Unfreeze last block
    param.requires_grad = True

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

# Training function
def train(model, train_loader, val_loader, num_epochs=10):
    best_loss = float("inf")
    train_losses, val_losses = [], []
    train_accuracies = []
    val_accuracies = []


    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == y).sum().item()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader)

        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] → Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_finetuned_model.pth")
            print(f"✅ Best fine-tuned model saved with loss: {best_loss:.4f}")

# Evaluation function
def evaluate(model, loader):
    model.eval()
    total_loss, y_true, y_pred, y_probs = 0, [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_probs)  # Compute AUC-ROC

    print(f"Validation Metrics → Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")

    return avg_loss, acc

# Train model
train(model, train_loader, val_loader, num_epochs=10)
