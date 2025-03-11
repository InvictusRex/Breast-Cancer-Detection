import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class
class ClassificationDataset(Dataset):
    def __init__(self, root_dir, transform, label_map):
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = label_map
        self.samples = []
        for label_name, _ in label_map.items():  # FIXED: Using label_name (string)
            class_dir = os.path.join(root_dir, label_name)  
            self.samples.extend([(os.path.join(class_dir, img), label_map[label_name]) 
                                for img in os.listdir(class_dir) if img.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(int(label), dtype=torch.float32)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load validation dataset
data_dir = r"E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\BreaKHis_v1\structured"
label_map = {"benign": 0, "malignant": 1}

val_dataset = ClassificationDataset(os.path.join(data_dir, "val"), transform, label_map)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Evaluation function
def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss, y_true, y_pred, y_probs = 0, [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze()
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_probs)

    return avg_loss, accuracy, precision, recall, f1, auc_roc

# -------------------- SimCLR-Finetuned Model --------------------
class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()  # Remove final FC layer
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

# Load SimCLR model
simclr_model = SimCLR().to(device)
simclr_model.load_state_dict(torch.load(r"E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\Task 2 - SimCLR Implementation\src\best_finetuned_model.pth", map_location=device))
simclr_model.eval()

# Evaluate SimCLR model
simclr_results = evaluate_model(simclr_model, val_loader, criterion)
print(f"\nSimCLR Fine-Tuned Model Results:\nLoss: {simclr_results[0]:.4f}, Accuracy: {simclr_results[1]:.4f}, Precision: {simclr_results[2]:.4f}, Recall: {simclr_results[3]:.4f}, F1-Score: {simclr_results[4]:.4f}, AUC-ROC: {simclr_results[5]:.4f}\n")

# -------------------- Pretrained ResNet-18 --------------------
class PretrainedResNet18(nn.Module):
    def __init__(self):
        super(PretrainedResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)  # Binary classification

    def forward(self, x):
        return self.model(x)

# Load pretrained ResNet-18 model
pretrained_model = PretrainedResNet18().to(device)
pretrained_model.eval()

# Evaluate Pretrained ResNet-18
pretrained_results = evaluate_model(pretrained_model, val_loader, criterion)
print(f"\nPretrained ResNet-18 Results:\nLoss: {pretrained_results[0]:.4f}, Accuracy: {pretrained_results[1]:.4f}, Precision: {pretrained_results[2]:.4f}, Recall: {pretrained_results[3]:.4f}, F1-Score: {pretrained_results[4]:.4f}, AUC-ROC: {pretrained_results[5]:.4f}\n")

# -------------------- Final Comparison --------------------
print("\n### Model Comparison ###")
print(f"Metric            | SimCLR Fine-Tuned  | Pretrained ResNet-18")
print(f"----------------------------------------------------------")
print(f"Loss              | {simclr_results[0]:.4f}           | {pretrained_results[0]:.4f}")
print(f"Accuracy          | {simclr_results[1]:.4f}           | {pretrained_results[1]:.4f}")
print(f"Precision         | {simclr_results[2]:.4f}           | {pretrained_results[2]:.4f}")
print(f"Recall            | {simclr_results[3]:.4f}           | {pretrained_results[3]:.4f}")
print(f"F1-Score          | {simclr_results[4]:.4f}           | {pretrained_results[4]:.4f}")
print(f"AUC-ROC           | {simclr_results[5]:.4f}           | {pretrained_results[5]:.4f}")
