from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Build the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
cnn_model.summary()

# Train the CNN model
history = cnn_model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size
)

# Evaluate the model on validation data
val_labels = val_data.classes  # True labels
val_preds = cnn_model.predict(val_data, steps=val_data.samples // val_data.batch_size + 1).ravel()
val_preds_binary = (val_preds > 0.5).astype(int)  # Convert probabilities to binary predictions

# Classification report
print("\nClassification Report:")
print(classification_report(val_labels, val_preds_binary, target_names=['Benign', 'Malignant']))

# AUC-ROC
auc_roc = roc_auc_score(val_labels, val_preds)
print(f"\nAUC-ROC: {auc_roc:.4f}")

# Save the trained model
cnn_model.save('cnn_model_breast_cancer.h5')
print("\nModel saved as 'cnn_model_breast_cancer.h5'")

# Visualize training history
def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(history)
