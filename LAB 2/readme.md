
# Fashion-MNIST Image Classification using Multi-Layer Perceptron (MLP) with Hyperparameter Optimization

## Overview

This project implements an **image classification system** for the **Fashion-MNIST** dataset using a **Multi-Layer Perceptron (MLP)**. The objective is to classify grayscale images of clothing items into one of ten categories and improve classification performance through **hyperparameter optimization**. The project compares the performance of a baseline MLP model with an optimized MLP model obtained using **Randomized Search Cross-Validation**.

---

# Objective

The objectives of this project are:

* Build a baseline Multi-Layer Perceptron (MLP) classifier for Fashion-MNIST.
* Preprocess and normalize image data before training.
* Analyze dataset characteristics and class distribution.
* Train the model and monitor training and validation performance.
* Optimize hyperparameters using Randomized Search.
* Compare baseline and optimized models.
* Evaluate the final model using accuracy, loss curves, and confusion matrix.

---

# Dataset Description

## Dataset

**Fashion-MNIST**

Fashion-MNIST is a benchmark dataset introduced by Zalando Research as a replacement for the original MNIST handwritten digit dataset. It consists of grayscale images representing various fashion products.

| Property                 |          Value |
| ------------------------ | -------------: |
| Dataset                  |  Fashion-MNIST |
| Number of Classes        |             10 |
| Training Images          |         60,000 |
| Test Images              |         10,000 |
| Image Size               | 28 × 28 pixels |
| Image Type               |      Grayscale |
| Total Features per Image |            784 |

---

## Class Labels

| Label | Class       |
| ----: | ----------- |
|     0 | T-shirt/top |
|     1 | Trouser     |
|     2 | Pullover    |
|     3 | Dress       |
|     4 | Coat        |
|     5 | Sandal      |
|     6 | Shirt       |
|     7 | Sneaker     |
|     8 | Bag         |
|     9 | Ankle boot  |

---

## Dataset Characteristics

| Characteristic      | Description                   |
| ------------------- | ----------------------------- |
| Balanced Dataset    | Yes (6,000 images per class)  |
| Image Format        | 28 × 28 grayscale             |
| Classification Type | Multi-class                   |
| Number of Classes   | 10                            |
| Input Dimension     | 784 features after flattening |
| Output              | One of 10 clothing categories |

---

# Project Workflow

```
Fashion-MNIST Dataset
          │
          ▼
Load Dataset
          │
          ▼
Data Preprocessing
• Normalization
• Flatten Images
          │
          ▼
Exploratory Data Analysis
• Class Distribution
• Sample Images
          │
          ▼
Baseline MLP Training
          │
          ▼
Performance Evaluation
• Accuracy
• Loss
• Confusion Matrix
          │
          ▼
Hyperparameter Optimization
(Randomized Search CV)
          │
          ▼
Optimized MLP Model
          │
          ▼
Performance Comparison
```

---

# Contents of the Jupyter Notebook (DL_LAB2.ipynb)

The notebook consists of the following sections:

| Section                     | Description                                                                      |
| --------------------------- | -------------------------------------------------------------------------------- |
| Import Libraries            | Import TensorFlow, Scikit-learn, NumPy, Matplotlib, and supporting libraries.    |
| Dataset Loading             | Load the Fashion-MNIST training and testing datasets.                            |
| Data Preprocessing          | Normalize pixel values and flatten images for MLP input.                         |
| Exploratory Data Analysis   | Display sample images and analyze class distribution.                            |
| Baseline Model              | Construct and train a baseline Multi-Layer Perceptron.                           |
| Training Visualization      | Plot training/validation accuracy and loss curves.                               |
| Model Evaluation            | Generate confusion matrix and evaluate classification performance.               |
| Hyperparameter Optimization | Perform Randomized Search Cross-Validation to identify the best hyperparameters. |
| Optimized Model             | Train and evaluate the optimized MLP model.                                      |
| Performance Comparison      | Compare baseline and optimized model accuracies.                                 |

---

# Model Configuration

## Baseline Model

| Parameter      | Value                           |
| -------------- | ------------------------------- |
| Model          | Multi-Layer Perceptron (MLP)    |
| Input Size     | 784                             |
| Output Classes | 10                              |
| Activation     | ReLU                            |
| Optimizer      | Adam                            |
| Loss Function  | Sparse Categorical Crossentropy |
| Epochs         | 20                              |

---

## Hyperparameter Optimization

Optimization was performed using **Randomized Search Cross-Validation (RandomizedSearchCV)**.

The search explored combinations of:

* Hidden layer size
* Learning rate
* Activation function
* Regularization parameter (Alpha)
* Optimizer
* Maximum iterations

The best-performing configuration was selected based on **mean cross-validation accuracy**.

---

# Results

## Baseline Model Performance

| Metric                   |     Value |
| ------------------------ | --------: |
| Test Accuracy            | **87.3%** |
| Final Training Accuracy  |    ~93.2% |
| Best Validation Accuracy |    ~89.1% |
| Final Training Loss      |     ~0.18 |
| Lowest Validation Loss   |    ~0.316 |

---

## Optimized Model Performance

| Metric        |     Value |
| ------------- | --------: |
| Test Accuracy | **89.4%** |
| Improvement   | **+2.1%** |

---

## Performance Comparison

| Model         | Test Accuracy |
| ------------- | ------------: |
| Baseline MLP  |     **87.3%** |
| Optimized MLP |     **89.4%** |

---

# Visualizations Included

The notebook contains the following visualizations:

| Figure                          | Purpose                                |
| ------------------------------- | -------------------------------------- |
| Training Set Class Distribution | Verify balanced dataset                |
| Sample Fashion-MNIST Images     | Visual inspection of dataset           |
| Training Accuracy vs Epoch      | Monitor learning progress              |
| Validation Accuracy vs Epoch    | Evaluate generalization                |
| Training Loss vs Epoch          | Observe convergence                    |
| Validation Loss vs Epoch        | Detect overfitting                     |
| Confusion Matrix                | Analyze class-wise predictions         |
| Hyperparameter Search Results   | Compare candidate configurations       |
| Baseline vs Optimized Accuracy  | Measure improvement after optimization |

---

# Key Observations

* The Fashion-MNIST dataset is perfectly balanced, reducing the risk of class bias.
* Training accuracy steadily increased from approximately **82%** to **93%**, indicating successful learning.
* Validation accuracy stabilized around **89%**, demonstrating good generalization.
* Training loss consistently decreased, confirming effective optimization.
* Validation loss began increasing slightly after later epochs, suggesting mild overfitting.
* The confusion matrix showed excellent performance for **Trouser**, **Sandal**, **Sneaker**, **Bag**, and **Ankle boot**.
* Most classification errors occurred between visually similar clothing categories such as **Shirt**, **Pullover**, **Coat**, and **T-shirt/top**.
* Hyperparameter optimization improved the overall classification accuracy by approximately **2.1%**.

---

# Technologies Used

| Tool               | Purpose                                  |
| ------------------ | ---------------------------------------- |
| Python             | Programming Language                     |
| TensorFlow / Keras | Deep Learning Framework                  |
| Scikit-learn       | Machine Learning & Hyperparameter Search |
| NumPy              | Numerical Computation                    |
| Matplotlib         | Data Visualization                       |
| Seaborn            | Confusion Matrix Visualization           |
| Jupyter Notebook   | Development Environment                  |

---

# Conclusion

This project demonstrates the effectiveness of **Multi-Layer Perceptrons (MLPs)** for multi-class image classification using the Fashion-MNIST dataset. The baseline model achieved strong performance with a test accuracy of **87.3%**, while hyperparameter optimization using **Randomized Search Cross-Validation** further improved the accuracy to **89.4%**. Training and validation curves indicate stable convergence with only mild overfitting during later epochs. The confusion matrix reveals that the model accurately classifies visually distinct categories such as footwear and bags, while errors mainly occur among similar upper-body garments. Overall, the optimized MLP provides a reliable and efficient solution for Fashion-MNIST image classification, highlighting the importance of systematic hyperparameter tuning in enhancing model performance.
