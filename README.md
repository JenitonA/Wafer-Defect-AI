# Wafer-Defect-AI

## Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify wafer defects.  
The goal is to train a deep learning model that can automatically detect patterns in wafer images and classify them into defect categories.

## Why This is Important
In semiconductor manufacturing, even tiny wafer defects can lead to significant yield loss and higher production costs.  
Automating defect detection with **deep learning improves accuracy, speed, and scalability compared to manual inspection.**  
This helps engineers reduce waste, **improve process reliability, and ensure higher-quality semiconductor devices.**

## Features
- Data preprocessing (loading, reshaping, and normalizing wafer images)  
- **Convolutional Neural Network (CNN)** architecture with convolution, pooling, and dense layers  
- Model training with TensorFlow/Keras  
- Accuracy and loss monitoring during training  
- Model evaluation on unseen test data  
- Visualization of learning curves  

## Requirements
Make sure the following packages are installed:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
