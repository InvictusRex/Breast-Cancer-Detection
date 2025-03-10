from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import pandas as pd

# Preprocessing: Define train_data and val_data (ensure preprocessing matches your training)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_path = r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\BreaKHis_v1\flattened_histology_slides'

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # Use the same split as training
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Function to evaluate a model and return metrics
def evaluate_model(model, val_data, model_name):
    # Get true labels and predictions
    val_labels = val_data.classes
    val_preds = model.predict(val_data, steps=val_data.samples // val_data.batch_size + 1).ravel()
    val_preds_binary = (val_preds > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Compute metrics
    accuracy = accuracy_score(val_labels, val_preds_binary)
    f1 = f1_score(val_labels, val_preds_binary)
    auc_roc = roc_auc_score(val_labels, val_preds)

    # Print classification report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(val_labels, val_preds_binary, target_names=['Benign', 'Malignant']))
    print(f"{model_name} AUC-ROC: {auc_roc:.4f}")

    return accuracy, f1, auc_roc

# Load the saved models
cnn_model = load_model(r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\Breast_Cancer_CNN\venv\src\cnn_model_breast_cancer.h5')
transfer_model = load_model(r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\Breast_Cancer_CNN\venv\src\transfer_model_resnet50_breast_cancer.h5')

# Evaluate Basic CNN
accuracy_cnn, f1_score_cnn, auc_roc_cnn = evaluate_model(cnn_model, val_data, "Basic CNN")

# Evaluate ResNet50 (Fine-Tuned)
accuracy_finetune, f1_score_finetune, auc_roc_finetune = evaluate_model(transfer_model, val_data, "ResNet50 (Fine-Tuned)")

# Create a DataFrame for the comparison table
metrics_data = {
    "Metric": ["Accuracy", "F1-Score", "AUC-ROC"],
    "Basic CNN": [accuracy_cnn, f1_score_cnn, auc_roc_cnn],
    "ResNet50 (Fine-Tuned)": [accuracy_finetune, f1_score_finetune, auc_roc_finetune]
}

metrics_df = pd.DataFrame(metrics_data)

# Display the comparison table
print("\nModel Performance Comparison:")
print(metrics_df)
