import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Define ClassificationDataset
class ClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_map=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = label_map
        self.image_paths = []
        self.labels = []

        # Collect image paths and labels
        for label_name, label in label_map.items():
            class_dir = os.path.join(root_dir, label_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, label

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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLR().to(device)

# Load trained weights
model.load_state_dict(torch.load(
    "E:/Work Files/Semester Notes/Semester 4/Artificial Intelligence/Task 2 - SimCLR Implementation/src/best_finetuned_model.pth",
    map_location=device
))
model.eval()  # Set model to evaluation mode

# Define dataset paths
data_dir = "E:/Work Files/Semester Notes/Semester 4/Artificial Intelligence/BreaKHis_v1/structured"
label_map = {"benign": 0, "malignant": 1}  # Fixing label_map to use class names as keys

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load validation dataset
val_dataset = ClassificationDataset(os.path.join(data_dir, "val"), transform, label_map)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define loss function
criterion = nn.BCEWithLogitsLoss()

# Define evaluation function
def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss, y_true, y_pred, y_probs = 0, [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze()  # Ensure logits shape matches labels
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
    
    print("\nValidation Metrics:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}\n")
    
    return avg_loss, accuracy, precision, recall, f1, auc_roc

# Run evaluation
val_loss, val_acc, precision, recall, f1, auc_roc = evaluate_model(model, val_loader, criterion)
