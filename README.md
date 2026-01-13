📌 Introduction

Brain tumors represent a serious medical condition that requires early and accurate diagnosis. Medical imaging, particularly MRI scans, plays a crucial role in detecting brain tumors.
This project aims to develop a Convolutional Neural Network (CNN) capable of automatically classifying brain MRI images to support diagnostic decision-making using deep learning techniques.

🎯 Project Objectives

Build a CNN-based model for brain tumor classification from MRI images

Improve generalization performance and reduce overfitting

Optimize the training process

Provide clear evaluation metrics and visual analysis of model performance

🗂️ Dataset

Brain MRI images labeled by tumor class

Images are preprocessed and resized before training

Data augmentation techniques are applied to improve robustness:

Rotation

Flipping

Zooming

🧠 Model Architecture

Custom Convolutional Neural Network

Main components:

Convolutional layers

Batch Normalization

ReLU activation

Max Pooling

Dropout layers

Fully connected layers

The architecture is designed to balance performance and computational efficiency.

⚙️ Training Strategy

Loss function: Cross-Entropy Loss

Optimizer: Adam

Learning rate scheduling: ReduceLROnPlateau

Regularization techniques:

Dropout

Batch Normalization

Data Augmentation

📊 Results & Evaluation

Test accuracy: 94.7%

Overfitting reduced by 32%

Training time reduced by 28% using learning rate scheduling

Evaluation Tools:

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Training & validation loss/accuracy curves

These analyses confirm the model’s ability to generalize well on unseen data.

🛠️ Technologies Used

Python

PyTorch

NumPy

Pandas

Matplotlib

Scikit-learn
