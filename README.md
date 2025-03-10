# Breast Cancer Detection

This repository contains an implementation of a **breast cancer detection model** using two approaches:

1. **Traditional CNN Training**: A basic CNN model is trained and compared with pre-trained models such as ResNet50, MobileNet, and VGG18 for binary classification (benign vs. malignant).
2. **Self-Supervised Learning (SSL) with SimCLR**: A SimCLR-based model is pretrained using contrastive learning and then fine-tuned for binary classification.

This project serves as a **comparative study** between the **supervised CNN approach** and **self-supervised learning with SimCLR** to analyze their effectiveness in detecting breast cancer.

## Dataset

The **BreaKHis v1 dataset** is used, which contains histopathological images of breast cancer tissue. The dataset is structured into two classes:

- **Benign** (non-cancerous)
- **Malignant** (cancerous)

(Link to Dataset: [BreaKHis Dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/))

## Approach

### 1. Basic CNN Training and Pretrained Model Comparison
- A **custom CNN** is trained on the dataset for binary classification.
- The CNN model's performance is compared against **pretrained models** such as **ResNet50, MobileNet, and VGG18**.
- The models are trained with standard **cross-entropy loss** and **Adam optimizer**.
- Performance metrics such as accuracy, precision, recall, and F1-score are used for evaluation.
- The goal is to measure the impact of **transfer learning** and fine-tuning pre-trained networks on the classification task.

### 2. Self-Supervised Learning with SimCLR
- The **SimCLR framework** is used for pretraining a feature extractor without labels.
- Images are augmented using random cropping, color jittering, and Gaussian blur before being fed into the model.
- A **contrastive loss function** is applied to maximize similarity between augmented views of the same image and minimize similarity between different images.
- The pre-trained SimCLR model is later **fine-tuned** using labeled data for binary classification.
- The objective is to compare **self-supervised learning (SSL)** with fully supervised CNN training and measure if it improves generalization and feature representation.

## Results

- **Supervised CNN training** shows strong performance but relies heavily on labeled data.
- **Pretrained models (ResNet50, MobileNet, VGG18, etc.)** achieve higher accuracy due to feature transfer from large-scale datasets.
- **SimCLR-based training** improves feature extraction and generalization, leading to better classification accuracy in limited-data scenarios.
- The study highlights the trade-offs between supervised learning and self-supervised learning in medical imaging applications.

## Running TensorFlow/PyTorch on GPU Locally

To efficiently train deep learning models, we set up GPU acceleration using the following configurations:

### 1. TensorFlow-GPU Support and Compatibility
- Since TensorFlow-GPU support for Windows 10 was discontinued after version **2.10.0**, we ensured compatibility by using **Python 3.9.11** and **TensorFlow 2.10.0**.
- This allows TensorFlow to leverage GPU acceleration efficiently without encountering compatibility issues.

### 2. CUDA and cuDNN Setup
- **CUDA v11.2** and **cuDNN v8.1** are installed to enable GPU acceleration.
- These libraries provide the necessary interface for running deep learning operations on **NVIDIA GPUs**, significantly improving training speed.

### 3. TensorRT for Optimized Inference
- **TensorRT** is installed to optimize inference, enabling faster execution of deep learning models without compromising accuracy.
- This helps in reducing latency during model deployment.

### 4. PyTorch with CUDA
- **PyTorch** is configured to use CUDA for GPU acceleration.
- This ensures that SimCLR training benefits from hardware acceleration, leading to reduced training time and improved performance.

## Trained Models

Pretrained models and trained checkpoints can be found in the following Google Drive folder:
[Trained Models](https://drive.google.com/drive/folders/1f_Bnj2U1_SWUxfh6RAW5HQSkCn-agVeL?usp=sharing)

## Future Work

- Experiment with different contrastive learning techniques.
- Apply the model to other medical imaging datasets.
- Optimize data augmentation strategies.

## License

This project is open-source under the **MIT License**.



