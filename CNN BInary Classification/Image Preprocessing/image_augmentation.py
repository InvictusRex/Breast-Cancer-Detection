from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the flattened dataset
data_path = r'E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\BreaKHis_v1\flattened_histology_slides'

# ImageDataGenerator for preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Rotate images up to 20 degrees
    width_shift_range=0.2,  # Horizontally shift images by 20% of width
    height_shift_range=0.2,  # Vertically shift images by 20% of height
    zoom_range=0.2,  # Zoom in/out by 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    validation_split=0.2  # Reserve 20% of data for validation
)

# Load training and validation datasets
train_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='training',  # Use for training
    shuffle=True  # Shuffle data for training
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='validation',  # Use for validation
    shuffle=False  # Do not shuffle validation data
)
