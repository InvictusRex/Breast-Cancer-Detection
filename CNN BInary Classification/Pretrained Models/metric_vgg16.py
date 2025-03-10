from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocessing: Define val_data
data_path = r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\BreaKHis_v1\flattened_histology_slides'

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # Ensure the same split as during training
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
    val_labels = val_data.classes
    val_preds = model.predict(val_data, steps=val_data.samples // val_data.batch_size + 1).ravel()
    val_preds_binary = (val_preds > 0.5).astype(int)

    accuracy = accuracy_score(val_labels, val_preds_binary)
    f1 = f1_score(val_labels, val_preds_binary)
    auc_roc = roc_auc_score(val_labels, val_preds)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(val_labels, val_preds_binary, target_names=['Benign', 'Malignant']))
    print(f"{model_name} AUC-ROC: {auc_roc:.4f}")

    return accuracy, f1, auc_roc

# Load saved models
cnn_model = load_model(r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\Breast_Cancer_CNN\venv\src\cnn_model_breast_cancer.h5')
resnet_model = load_model(r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\Breast_Cancer_CNN\venv\src\transfer_model_resnet50_breast_cancer.h5')
vgg_model = load_model(r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\Breast_Cancer_CNN\venv\src\transfer_model_vgg16_breast_cancer.h5')

# Evaluate Basic CNN
accuracy_cnn, f1_score_cnn, auc_roc_cnn = evaluate_model(cnn_model, val_data, "Basic CNN")

# Evaluate ResNet50 (Fine-Tuned)
accuracy_resnet, f1_score_resnet, auc_roc_resnet = evaluate_model(resnet_model, val_data, "ResNet50 (Fine-Tuned)")

# Evaluate VGG16 (Fine-Tuned)
accuracy_vgg, f1_score_vgg, auc_roc_vgg = evaluate_model(vgg_model, val_data, "VGG16 (Fine-Tuned)")

# Create a comparison table
metrics_data = {
    "Metric": ["Accuracy", "F1-Score", "AUC-ROC"],
    "Basic CNN": [accuracy_cnn, f1_score_cnn, auc_roc_cnn],
    "ResNet50 (Fine-Tuned)": [accuracy_resnet, f1_score_resnet, auc_roc_resnet],
    "VGG16 (Fine-Tuned)": [accuracy_vgg, f1_score_vgg, auc_roc_vgg]
}

metrics_df = pd.DataFrame(metrics_data)

# Display the comparison table
print("\nModel Performance Comparison:")
print(metrics_df)
