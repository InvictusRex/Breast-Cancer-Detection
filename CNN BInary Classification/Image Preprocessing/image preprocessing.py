import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure TensorFlow detects and uses the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs available: {len(physical_devices)}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected. Please check your TensorFlow-GPU installation.")

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

# GPU confirmation
print(f"Using GPU: {tf.test.is_gpu_available()}")
