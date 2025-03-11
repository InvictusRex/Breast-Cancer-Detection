import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from simclr_model import SimCLR  # Ensure this file contains the correct model definition

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = SimCLR().to(device)  # Ensure this matches the training architecture
model.load_state_dict(torch.load("E:\\Work Files\\Semester Notes\\Semester 4\\Artificial Intelligence\\Task 2 - SimCLR Implementation\\src\\best_finetuned_model.pth"))
model.eval()

# Define transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size based on training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load validation dataset
val_path = "E:\\Work Files\\Semester Notes\\Semester 4\\Artificial Intelligence\\BreaKHis_v1\\structured\\val"
val_dataset = ImageFolder(root=val_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def evaluate_model(model, val_loader):
    """Evaluates the model on the validation dataset."""
    correct, total = 0, 0
    all_preds, all_labels = []
    val_loss = 0  # Placeholder if loss calculation is required

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = correct / total

    # Additional metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_preds)

    print("\nValidation Metrics:")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}\n")

    return val_loss, accuracy, precision, recall, f1, auc_roc

# Run evaluation
val_loss, val_acc, precision, recall, f1, auc_roc = evaluate_model(model, val_loader)

# Generate and save performance graph
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
values = [val_acc, precision, recall, f1, auc_roc]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.ylim(0, 1)  # Metrics are between 0 and 1
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("Validation Performance Metrics")
plt.savefig("performance_metrics.png")
plt.show()
