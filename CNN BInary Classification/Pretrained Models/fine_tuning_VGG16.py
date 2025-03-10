from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing: Define train_data and val_data
data_path = r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\BreaKHis_v1\flattened_histology_slides'

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # Use the same split as training
)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Load pretrained VGG16 model
base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_vgg.trainable = False  # Freeze base model layers

# Add custom classification layers
vgg_model = Sequential([
    base_model_vgg,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the VGG16 transfer learning model
vgg_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train VGG16 with frozen base layers
history_vgg = vgg_model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=5,  # Adjust based on your dataset
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size
)

# Fine-tune VGG16 by unfreezing some layers
base_model_vgg.trainable = True  # Unfreeze all layers (or specific ones as needed)

# Recompile with a lower learning rate for fine-tuning
vgg_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the VGG16 model
history_vgg_finetune = vgg_model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=5,  # Fine-tuning phase
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size
)

# Save the fine-tuned VGG16 model
vgg_model.save('transfer_model_vgg16_breast_cancer.h5')
print("\nFine-tuned VGG16 model saved as 'transfer_model_vgg16_breast_cancer.h5'")

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

# Load previously saved models
cnn_model = load_model('cnn_model.h5')
resnet_model = load_model('transfer_model_resnet50_breast_cancer.h5')

# Evaluate models
accuracy_cnn, f1_score_cnn, auc_roc_cnn = evaluate_model(cnn_model, val_data, "Basic CNN")
accuracy_resnet, f1_score_resnet, auc_roc_resnet = evaluate_model(resnet_model, val_data, "ResNet50 (Fine-Tuned)")
accuracy_vgg, f1_score_vgg, auc_roc_vgg = evaluate_model(vgg_model, val_data, "VGG16 (Fine-Tuned)")

# Create comparison table
metrics_data = {
    "Metric": ["Accuracy", "F1-Score", "AUC-ROC"],
    "Basic CNN": [accuracy_cnn, f1_score_cnn, auc_roc_cnn],
    "ResNet50 (Fine-Tuned)": [accuracy_resnet, f1_score_resnet, auc_roc_resnet],
    "VGG16 (Fine-Tuned)": [accuracy_vgg, f1_score_vgg, auc_roc_vgg]
}

metrics_df = pd.DataFrame(metrics_data)

# Display comparison table
print("\nModel Performance Comparison:")
print(metrics_df)

# Visualize training history
def plot_training_history(history, history_finetune, title):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy (Frozen)')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy (Frozen)')
    plt.plot(history_finetune.history['accuracy'], label='Train Accuracy (Fine-Tune)')
    plt.plot(history_finetune.history['val_accuracy'], label='Val Accuracy (Fine-Tune)')
    plt.title(f'Training and Validation Accuracy - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss (Frozen)')
    plt.plot(history.history['val_loss'], label='Val Loss (Frozen)')
    plt.plot(history_finetune.history['loss'], label='Train Loss (Fine-Tune)')
    plt.plot(history_finetune.history['val_loss'], label='Val Loss (Fine-Tune)')
    plt.title(f'Training and Validation Loss - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot VGG16 training history
plot_training_history(history_vgg, history_vgg_finetune, "VGG16")
