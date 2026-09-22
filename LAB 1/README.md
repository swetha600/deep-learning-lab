

## Objective

The objective of this project is to implement a **Single Layer Perceptron (SLP)** from scratch using Python for the binary classification of authentic and forged banknotes. The project aims to understand the perceptron learning algorithm, including weight initialization, bias updates, prediction using the Step Activation Function, and iterative learning through weight updates. The project also includes exploratory data analysis, feature normalization, model evaluation, visualization of the learning process, comparison with Scikit-learn's implementation, and the study of different learning rates and activation functions.

---

## Dataset Description

The project uses the **Banknote Authentication Dataset** from the **UCI Machine Learning Repository**. The dataset consists of **1,372 banknote samples**, each represented by four statistical features extracted from wavelet-transformed images. These features are **Variance, Skewness, Curtosis, and Entropy**, while the target variable classifies each banknote as either **Authentic (0)** or **Forged (1)**. Since the dataset is largely linearly separable, it is well suited for demonstrating the working of a Single Layer Perceptron.

---

## Contents of the Jupyter Notebook (.ipynb)

The notebook contains the complete implementation of the Single Layer Perceptron along with data preprocessing, exploratory data analysis, model training, and evaluation. The major sections include:

* Importing required libraries
* Loading and preprocessing the dataset
* Feature normalization and train-test splitting
* Exploratory Data Analysis

  * Histograms
  * Box Plots
  * Correlation Heatmap
  * Scatter Plots
* Implementation of the Single Layer Perceptron from scratch
* Model training using the Step Activation Function
* Performance evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
* Visualization of:

  * Training Error vs Epoch
  * Weight Evolution
  * Bias Evolution
  * Learning Rate Comparison
  * Decision Boundary
* Additional experiments including:

  * Step vs Sigmoid Activation Function
  * Scikit-learn Perceptron Comparison
  * XOR Problem
  * Effect of Feature Normalization

---

## Results

The implemented Single Layer Perceptron successfully learned a linear decision boundary and achieved excellent performance on the Banknote Authentication dataset. Feature normalization improved training stability and convergence, while the exploratory data analysis confirmed that the extracted statistical features provide good class separability.

| **Evaluation Metric** | **Value**  |
| --------------------- | ---------- |
| Accuracy              | **96.36%** |
| Precision             | **93.08%** |
| Recall                | **99.18%** |
| F1-Score              | **96.03%** |

The model produced only a few misclassifications, as observed from the confusion matrix, and the training error gradually decreased until convergence. The learning rate comparison showed similar convergence for all three learning rates due to feature normalization and the nearly linearly separable nature of the dataset.

---

## Conclusion

The project successfully demonstrates the implementation of a **Single Layer Perceptron** from scratch for binary classification. The model effectively classified authentic and forged banknotes with an accuracy of approximately **96.36%**, confirming that the dataset is largely linearly separable. The experiment also highlights the importance of feature normalization, appropriate learning rate selection, and exploratory data analysis in achieving stable convergence and high classification performance. Overall, this project provides a solid understanding of the perceptron learning algorithm and serves as a strong foundation for studying more advanced neural network models such as **Multilayer Perceptrons (MLPs)** and deep learning architectures.
