from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the flattened dataset
data_path = r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\BreaKHis_v1\flattened_histology_slides'

# ImageDataGenerator for preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Rotate images
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2,  # Shift images vertically
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True,  # Flip images horizontally
    validation_split=0.2  # Split into training and validation sets
)

# Load training and validation datasets
train_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),  # Resize images to match input size
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='training',  # Specify training subset
    shuffle=True  # Shuffle data for training
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),  # Resize images to match input size
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='validation',  # Specify validation subset
    shuffle=False  # Do not shuffle validation data
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
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
cnn_model.summary()

# Train the model
cnn_model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=10,  # You can adjust the number of epochs
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size
)
