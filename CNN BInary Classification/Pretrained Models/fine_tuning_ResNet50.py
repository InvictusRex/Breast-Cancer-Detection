from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Preprocessing: Define train_data and val_data
data_path = r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\BreaKHis_v1\flattened_histology_slides'

datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Rotate images up to 20 degrees
    width_shift_range=0.2,  # Horizontally shift images by 20% of width
    height_shift_range=0.2,  # Vertically shift images by 20% of height
    zoom_range=0.2,  # Zoom in/out by 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    validation_split=0.2  # Reserve 20% of data for validation
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

# Load the pretrained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model layers initially

# Add custom classification layers
transfer_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Reduce feature maps to a single vector
    Dense(128, activation='relu'),  # Fully connected layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the transfer learning model
transfer_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
transfer_model.summary()

# Train the transfer learning model (frozen base layers)
history_transfer = transfer_model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=5,  # Adjust based on your dataset
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size
)

# Fine-tune the model by unfreezing some layers of the base model
base_model.trainable = True  # Unfreeze all layers (or specific layers if desired)

# Recompile with a lower learning rate
transfer_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the transfer learning model
history_finetune = transfer_model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=5,  # Fine-tuning phase
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size
)

# Evaluate the fine-tuned model
val_preds_transfer = transfer_model.predict(val_data, steps=val_data.samples // val_data.batch_size + 1).ravel()
val_preds_transfer_binary = (val_preds_transfer > 0.5).astype(int)

print("\nTransfer Learning Model Classification Report:")
print(classification_report(val_data.classes, val_preds_transfer_binary, target_names=['Benign', 'Malignant']))

auc_roc_transfer = roc_auc_score(val_data.classes, val_preds_transfer)
print(f"\nTransfer Learning Model AUC-ROC: {auc_roc_transfer:.4f}")

# Save the fine-tuned transfer learning model
transfer_model.save('transfer_model_resnet50_breast_cancer.h5')
print("\nFine-tuned transfer learning model saved as 'transfer_model_resnet50_breast_cancer.h5'")

# Visualize training and fine-tuning history
def plot_training_history(history, history_finetune):
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy (Frozen)')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy (Frozen)')
    plt.plot(history_finetune.history['accuracy'], label='Train Accuracy (Fine-tune)')
    plt.plot(history_finetune.history['val_accuracy'], label='Val Accuracy (Fine-tune)')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss (Frozen)')
    plt.plot(history.history['val_loss'], label='Val Loss (Frozen)')
    plt.plot(history_finetune.history['loss'], label='Train Loss (Fine-tune)')
    plt.plot(history_finetune.history['val_loss'], label='Val Loss (Fine-tune)')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(history_transfer, history_finetune)
