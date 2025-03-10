import os
import shutil
from sklearn.model_selection import train_test_split

# Define dataset paths
original_dataset = r"E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\BreaKHis_v1\flattened_histology_slides"
new_dataset = r"E:\Work Files\Semester Notes\Semester 4\Artificial Intelligence\BreaKHis_v1\structured"

# Define classes
classes = ["benign", "malignant"]

# Create new dataset structure
for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join(new_dataset, split, cls), exist_ok=True)

# Train-validation split ratio
train_ratio = 0.8  # 80% train, 20% validation

# Process each class
for cls in classes:
    class_path = os.path.join(original_dataset, cls)
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Split data into train and validation sets
    train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)

    # Copy images to the new structured dataset
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(new_dataset, "train", cls, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(new_dataset, "val", cls, img))

print("Dataset restructuring complete! ✅")
