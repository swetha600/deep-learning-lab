# CS3807 – Deep Learning Laboratory
## Experiment 3: Implementation of Convolutional Neural Networks (CNNs) for Image Classification

**Institution:** Shiv Nadar University Chennai
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V
**Subject Code:** CS3807 – Deep Learning Laboratory
**Academic Year:** 2026–27

---

## 1. Objective

To understand the working principle of Convolutional Neural Networks (CNNs) by implementing:
- The convolution operation and output-dimension calculations
- Feature map visualization
- Max pooling and average pooling
- CNN parameter calculation
- A complete CNN pipeline for image classification on CIFAR-10
- Model evaluation using standard classification metrics

---

## 2. Dataset Description

**Dataset:** CIFAR-10

| Property | Value |
|---|---|
| Training images | 50,000 |
| Testing images | 10,000 |
| Number of classes | 10 |
| Image size | 32 × 32 × 3 (RGB) |
| Classes | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Images per class (training) | 5,000 (perfectly balanced) |

Pixel values were normalized to the range [0, 1] prior to training. The dataset is well-balanced across all 10 classes, confirmed by the class distribution plot, meaning accuracy is a fair and unbiased evaluation metric for this task.

---

## 3. Contents of the Notebook (`Experiment_3_CNN_CIFAR10.ipynb`)

The notebook is organized into the following sections, matching the lab manual's task structure:

| Section | Description |
|---|---|
| **0. Setup and Imports** | Imports TensorFlow/Keras, NumPy, Matplotlib, seaborn, scikit-learn metrics; sets random seeds for reproducibility |
| **Task 1 – Load CIFAR-10** | Loads the dataset, prints shapes, displays 10 sample images, plots class distribution |
| **Task 2 – Convolution Layer** | Implements convolution and compares kernel sizes (3×3, 5×5, 7×7), recording resulting feature map sizes |
| **Task 3 – Hyperparameter Study** | Compares stride (1 vs 2) and padding (same vs valid), computing output dimensions for each combination |
| **Task 4 – Feature Map Visualization** | Visualizes 8 feature maps produced after the first convolution layer |
| **Task 5 – Pooling Comparison** | Trains a quick 5-epoch model to compare max pooling vs average pooling on test accuracy |
| **Task 6 – CNN Construction & Training** | Builds and trains the full CNN: `Input → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Dense → Softmax`, using Adam optimizer, 20 epochs, batch size 32 |
| **Task 7 – Evaluation** | Computes accuracy, precision, recall, F1-score, confusion matrix, and classification report on the test set |
| **Mandatory Plots** | Generates and saves all 8 required plots (sample images, class distribution, training/validation accuracy & loss, feature maps, confusion matrix) |
| **Discussion** | Answers the 6 conceptual questions from the lab manual (convolution vs FC layers, stride, padding, pooling, feature maps, parameter efficiency) |
| **Export utility** | Converts saved `.eps` plot files to `.png` and zips them for submission |

---

## 4. Results

### 4.1 Final Model Performance (Task 6/7 CNN)

| Metric | Value |
|---|---|
| Training Accuracy | 89.48% |
| Testing Accuracy | 66.14% |
| Precision (weighted) | 0.6585 |
| Recall (weighted) | 0.6614 |
| F1-score (weighted) | 0.6565 |
| Total Trainable Parameters | 545,098 |
| Total Training Time | 1632.28 sec (~27.2 min) |

### 4.2 Pooling Comparison (Task 5 — 5-epoch quick test)

| Pooling Type | Test Accuracy |
|---|---|
| Max Pooling | ~0.64 |
| Average Pooling | ~0.60 |

**Observation:** Max pooling outperformed average pooling, since it preserves the strongest local activation (sharper edge/texture signals) rather than diluting it through averaging.

### 4.3 Per-Class Performance Highlights (from Confusion Matrix)

| Class | Correctly Classified (out of ~1000) | Notes |
|---|---|---|
| Automobile | 815 | Best performing class |
| Ship | 814 | Very well classified |
| Horse | 779 | Strong performance |
| Truck | 765 | Strong performance |
| Frog | 744 | Good performance |
| Deer | 633 | Moderate performance |
| Dog | 553 | Frequently confused with cat |
| Bird | 532 | Frequently confused with deer |
| Airplane | 596 | Some confusion with ship |
| Cat | 383 | Weakest class; confused with dog and deer |

**Most common confusions:** cat ↔ dog (163 dogs → cat, 141 cats → dog), bird ↔ deer (137 birds → deer). Rigid, geometrically distinct objects (vehicles) were classified far more reliably than texture-heavy, pose-variable animals.

### 4.4 Training Behavior Summary

| Curve | Behavior |
|---|---|
| Training Accuracy | Rises steadily from ~0.51 to ~0.96 across 20 epochs |
| Validation Accuracy | Rises quickly to ~0.69–0.70 by epoch 2–3, then plateaus/oscillates (~0.67–0.70) with no further gains |
| Training Loss | Decreases smoothly and monotonically from ~1.38 to ~0.11 |
| Validation Loss | Decreases to a minimum of ~0.90 by epoch 2–4, then rises continuously to ~2.36 by epoch 19 |

**Key finding:** The widening gap between training and validation curves after epoch ~4 is a clear indicator of **overfitting** — the model kept reducing training error long after it stopped generalizing to unseen data.

---

## 5. Mandatory Plots Generated

1. Sample Images (`sample_images.png`)
2. Class Distribution (`class_distribution.png`)
3. Training Accuracy (`training_accuracy.png`)
4. Validation Accuracy (`validation_accuracy.png`)
5. Training Loss (`training_loss.png`)
6. Validation Loss (`validation_loss.png`)
7. Feature Maps after First Convolution Layer (`feature_maps.png`)
8. Confusion Matrix (`confusion_matrix.png`)

Each plot is accompanied by a short (2–3 line) inference in the notebook, in line with the lab manual's reporting requirement.

---

## 6. Conclusion

A CNN with two convolution–ReLU–max pooling blocks followed by a dense softmax classifier was successfully implemented and trained on CIFAR-10, achieving **89.48% training accuracy** and **66.14% test accuracy** (F1-score: 0.6565) with **545,098 trainable parameters**, in approximately 27 minutes of training.

The experiment met all its stated learning outcomes: the convolution operation, output-dimension formula, pooling mechanics, and parameter-counting were verified both numerically and empirically; feature maps were visualized and interpreted; and the trained classifier was evaluated with a complete set of standard metrics.

However, the ~23-point gap between training and test accuracy, combined with the diverging validation loss curve after epoch ~4, indicates the model **overfits substantially** on this relatively shallow architecture. The confusion matrix shows the network handles rigid, shape-distinct object classes (automobile, ship, horse, truck) well, but struggles to separate texture-heavy, pose-variable animal classes (cat, dog, bird, deer) due to limited representational depth.

**Recommended improvements for future iterations:**
1. **Early stopping** around epoch 2–4, where validation loss is minimized.
2. **Regularization** (dropout, L2 weight decay) to close the train/validation gap.
3. **Data augmentation** (random crops, flips, rotations) to improve generalization, especially for confused animal classes.
4. **Batch normalization** to stabilize and accelerate convergence.
5. **Increased network depth/filters** (e.g., 16 → 64 filters, as suggested in the lab manual's additional exercises), paired with the above regularization to prevent the added capacity from worsening overfitting.

Overall, the experiment successfully illustrates both the core strength of CNNs — automatic, parameter-efficient feature learning through weight sharing and local connectivity — and a common practical limitation, overfitting, providing a realistic foundation for understanding why regularization and architectural depth matter in real-world CNN design.

---

## 7. References

1. Goodfellow, I., Bengio, Y., & Courville, A. — *Deep Learning*
2. Bishop, C. M. — *Pattern Recognition and Machine Learning*
3. Haykin, S. — *Neural Networks and Learning Machines*
4. TensorFlow Documentation — https://www.tensorflow.org/
5. CIFAR-10 Dataset Documentation — https://www.cs.toronto.edu/~kriz/cifar.html
