

## Objective

The objective of this project is to implement a **Single Layer Perceptron (SLP)** from scratch using Python for binary classification of authentic and forged banknotes. The project aims to understand the perceptron learning algorithm, including weight and bias initialization, prediction using the Step Activation Function, weight updates based on the perceptron learning rule, and model convergence. Additionally, the project explores the dataset through exploratory data analysis, evaluates model performance using various classification metrics, compares different learning rates, studies the effect of feature normalization, and compares the custom implementation with Scikit-learn's Perceptron.

---

## Dataset Description

The project uses the **Banknote Authentication Dataset**, which is a binary classification dataset obtained from the **UCI Machine Learning Repository**. The dataset contains **1,372 banknote samples**, each represented by four statistical features extracted from wavelet-transformed images of genuine and forged banknotes.

**Features:**

* **Variance** – Measures the variance of the wavelet-transformed image.
* **Skewness** – Measures the asymmetry of the image distribution.
* **Curtosis** – Measures the peakedness of the image distribution.
* **Entropy** – Measures the randomness or complexity of the image.

**Target Variable:**

* **0** – Authentic Banknote
* **1** – Forged Banknote

The dataset is well suited for binary classification and is largely linearly separable, making it an ideal choice for implementing and studying the Single Layer Perceptron algorithm.

---

## Contents of the Jupyter Notebook (.ipynb)

The notebook contains the complete implementation and evaluation of the Single Layer Perceptron. It includes:

* Importing the required Python libraries.
* Loading and preprocessing the Banknote Authentication dataset.
* Exploratory Data Analysis (EDA), including:

  * Feature Histograms
  * Box Plots
  * Correlation Heatmap
  * Scatter Plots
* Feature normalization using StandardScaler.
* Splitting the dataset into training and testing sets.
* Implementing the Single Layer Perceptron from scratch.
* Training the perceptron using the Step Activation Function.
* Monitoring model convergence through:

  * Training Error vs Epoch
  * Weight Evolution
  * Bias Evolution
* Evaluating model performance using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
* Visualizing the learned Decision Boundary.
* Comparing learning rates (0.001, 0.01, and 0.1).
* Comparing the custom implementation with Scikit-learn's Perceptron.
* Discussing the Step and Sigmoid activation functions.
* Explaining why a Single Layer Perceptron cannot solve the XOR problem.
* Studying the effect of feature normalization on model convergence.

---

## Results

The implemented Single Layer Perceptron successfully learned a linear decision boundary for classifying authentic and forged banknotes. Exploratory Data Analysis showed that the extracted statistical features contain sufficient information for effective classification, with **Variance** being the most influential feature. Feature normalization improved the stability of training and enabled faster convergence.

The model achieved high classification performance with an **accuracy of approximately 96.36%**, along with high precision, recall, and F1-score. The confusion matrix showed only a small number of misclassified samples, indicating that the perceptron effectively distinguishes between authentic and forged banknotes. The learning curves demonstrated steady convergence, while the weight and bias evolution plots confirmed that the model stabilized after several training epochs. The learning rate comparison further showed that, after normalization, different learning rates produced similar convergence behaviour due to the nearly linearly separable nature of the dataset.

---

## Conclusion

This project successfully demonstrated the implementation of a **Single Layer Perceptron from scratch** for binary classification. The results show that a perceptron can achieve excellent performance on linearly separable datasets such as the Banknote Authentication dataset. The experiment also highlights the importance of feature normalization, proper learning rate selection, and exploratory data analysis in improving model performance and convergence. Furthermore, the additional experiments provided a better understanding of activation functions, the limitations of linear classifiers such as the inability to solve the XOR problem, and the advantages of optimized implementations available in Scikit-learn. Overall, this project provides a strong foundation for understanding perceptrons and serves as an introduction to more advanced neural network architectures such as Multilayer Perceptrons and deep learning models.
