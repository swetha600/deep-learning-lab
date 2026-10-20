# CS3807 – Deep Learning Laboratory

## Compiled Record of Experiments 1–9: From the Perceptron to Attention-Augmented Deep Networks

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow) ![Experiments](https://img.shields.io/badge/Experiments-9-green)

**Institution:** Shiv Nadar University Chennai
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V
**Subject Code:** CS3807 – Deep Learning Laboratory
**Academic Year:** 2026–27

---

## Table of Contents

1. [Overview](#1-overview)
2. [Experiment Index](#2-experiment-index)
3. [Learning Trajectory](#3-learning-trajectory)
4. [Experiment 1 – Single Layer Perceptron](#4-experiment-1--single-layer-perceptron)
5. [Experiment 2 – MLP with Hyperparameter Optimization](#5-experiment-2--mlp-with-hyperparameter-optimization)
6. [Experiment 3 – CNNs for Image Classification](#6-experiment-3--cnns-for-image-classification)
7. [Experiment 4 – Transfer Learning with Deep CNN Architectures](#7-experiment-4--transfer-learning-with-deep-cnn-architectures)
8. [Experiment 5 – CNN Training, Regularization, Optimization and Cross-Validation](#8-experiment-5--cnn-training-regularization-optimization-and-cross-validation)
9. [Experiment 6 – RNN, LSTM and GRU](#9-experiment-6--rnn-lstm-and-gru)
10. [Experiment 7 – Autoencoders](#10-experiment-7--autoencoders)
11. [Experiments 8 & 9 – Attention in a Pretrained ResNet50](#11-experiments-8--9--attention-in-a-pretrained-resnet50)
12. [Cross-Experiment Analysis](#12-cross-experiment-analysis)
13. [Repository Layout](#13-repository-layout)
14. [Technology Stack](#14-technology-stack)
15. [Reproducibility and Running the Experiments](#15-reproducibility-and-running-the-experiments)
16. [Known Limitations](#16-known-limitations)
17. [Overall Conclusion](#17-overall-conclusion)
18. [References](#18-references)

---

## 1. Overview

This repository compiles the full laboratory record for CS3807. The experiments follow a deliberate progression: a single neuron learned from scratch, multi-layer networks, convolutional networks, transfer learning, systematic training and regularization studies, sequence models, generative and representation-learning models, and finally attention mechanisms inserted into a pretrained backbone.

Every experiment contains:

- A Jupyter notebook with the complete implementation
- A set of mandatory plots, each followed by an inline inference (what the plot shows, the trend, why it occurs)
- A results section with quantitative metrics
- Answers to the lab manual's discussion questions
- Conclusions with stated limitations

## 2. Experiment Index

| # | Experiment | Dataset | Core Models / Techniques | Headline Result |
|---|---|---|---|---|
| 1 | Single Layer Perceptron from scratch | Banknote Authentication (UCI) | Perceptron, step vs. sigmoid, learning-rate study | 96.36% accuracy, F1 96.03% |
| 2 | MLP with hyperparameter optimization | Fashion-MNIST | MLP, `RandomizedSearchCV` | 87.3% → 89.4% test accuracy |
| 3 | CNNs for image classification | CIFAR-10 | Conv/Pool/Dense CNN, kernel/stride/padding/pooling study | 66.14% test accuracy (89.48% train) |
| 4 | Transfer learning with deep CNN architectures | CIFAR-10 | VGG16 frozen + fine-tuned, ResNet50 comparison | 65.90% → 77.73% after fine-tuning |
| 5 | CNN training, regularization, optimization, cross-validation | Oxford-IIIT Pet | MobileNetV2, 5-fold CV | Pipeline complete; results to be populated |
| 6 | RNN, LSTM, GRU, video understanding | UCI HAR, synthetic video, synthetic seq2seq | Gated RNNs, CNN–LSTM, encoder–decoder | Pipeline complete; results to be populated |
| 7 | Autoencoders (FC, CAE, DAE, VAE) | MNIST | Reconstruction, denoising, generative modeling | CAE SSIM 0.969 vs. FC-AE 0.735 |
| 8 & 9 | Attention type and position in ResNet50 | TensorFlow Flowers | SE, spatial, CBAM-style, ECA, self-attention, Grad-CAM | Self-attention at P3: 94.37% (baseline 93.28%) |

## 3. Learning Trajectory

| Stage | Experiments | Concept Introduced |
|---|---|---|
| Foundations | 1–2 | Learning rule, activation functions, MLPs, hyperparameter search |
| Spatial learning | 3–4 | Convolution, pooling, feature maps, transfer learning and fine-tuning |
| Training science | 5 | Initialization, regularization, BatchNorm, optimizers, cross-validation |
| Temporal learning | 6 | BPTT, vanishing gradients, LSTM/GRU gating, seq2seq |
| Representation and generation | 7 | Bottlenecks, denoising, latent spaces, reparameterization |
| Interpretability and attention | 8–9 | Channel/spatial/self-attention, position ablation, Grad-CAM |

---

## 4. Experiment 1 – Single Layer Perceptron

### 4.1 Objective

To implement a **Single Layer Perceptron (SLP)** from scratch for binary classification of authentic and forged banknotes, and to understand:

- Weight and bias initialization
- Prediction with the step activation function
- The perceptron learning rule and iterative weight updates
- Exploratory data analysis, feature normalization and model evaluation
- Visualization of the learning process
- Comparison with scikit-learn's implementation, learning-rate and activation-function studies

### 4.2 Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository, Banknote Authentication |
| Samples | 1,372 |
| Features | Variance, Skewness, Curtosis, Entropy (from wavelet-transformed images) |
| Target | Authentic (0) / Forged (1) |
| Separability | Largely linearly separable |

### 4.3 Notebook Structure

| Section | Description |
|---|---|
| Libraries & loading | Imports, dataset load and preprocessing |
| Normalization & split | Feature normalization and train–test split |
| Exploratory Data Analysis | Histograms, box plots, correlation heatmap, scatter plots |
| SLP implementation | Perceptron coded from scratch with step activation |
| Evaluation | Accuracy, Precision, Recall, F1-score, confusion matrix |
| Learning visualizations | Training error vs. epoch, weight evolution, bias evolution, learning-rate comparison, decision boundary |
| Additional experiments | Step vs. sigmoid activation, scikit-learn Perceptron comparison, XOR problem, effect of normalization |

### 4.4 Results

| Metric | Value |
|---|---|
| Accuracy | 96.36% |
| Precision | 93.08% |
| Recall | 99.18% |
| F1-score | 96.03% |

### 4.5 Observations

| Topic | Observation |
|---|---|
| Learning | The perceptron learned a linear decision boundary and training error decreased until convergence |
| Errors | Only a few misclassifications in the confusion matrix; recall (99.18%) is higher than precision (93.08%), so errors lean towards flagging genuine notes as forged |
| Normalization | Improved training stability and convergence |
| Learning rates | All three learning rates converged similarly, because of normalization and near-linear separability |
| Features | EDA confirmed good class separability of the extracted statistical features |

### 4.6 Conclusion

The SLP reaches about 96.36% accuracy, confirming the dataset is largely linearly separable. The experiment shows the importance of normalization, sensible learning-rate choice and EDA, and sets up the motivation for multilayer networks (the XOR exercise shows why a single layer is insufficient for non-linear problems).

---

## 5. Experiment 2 – MLP with Hyperparameter Optimization

### 5.1 Objective

To classify Fashion-MNIST clothing images with a Multi-Layer Perceptron and improve it via hyperparameter search:

- Build and train a baseline MLP
- Normalize and flatten image data
- Analyze class distribution and sample images
- Monitor training and validation behaviour
- Optimize hyperparameters with Randomized Search Cross-Validation
- Compare baseline and optimized models

### 5.2 Dataset

| Property | Value |
|---|---|
| Dataset | Fashion-MNIST (Zalando Research) |
| Classes | 10 |
| Training / test images | 60,000 / 10,000 |
| Image size | 28 × 28 grayscale (784 features after flattening) |
| Balance | Perfectly balanced, 6,000 images per class |

| Label | Class | Label | Class |
|---:|---|---:|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

### 5.3 Workflow

```
Load dataset → Normalize + flatten → EDA (class distribution, samples)
      → Baseline MLP → Evaluation (accuracy, loss, confusion matrix)
      → RandomizedSearchCV → Optimized MLP → Baseline vs. optimized comparison
```

### 5.4 Configuration

| Baseline Parameter | Value |
|---|---|
| Model | MLP |
| Input / output | 784 / 10 |
| Activation | ReLU |
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Epochs | 20 |

**Search space (RandomizedSearchCV):** hidden-layer size, learning rate, activation function, regularization alpha, optimizer, maximum iterations. The best configuration was selected by mean cross-validation accuracy.

### 5.5 Results

| Baseline Metric | Value |
|---|---|
| Test accuracy | 87.3% |
| Final training accuracy | ~93.2% |
| Best validation accuracy | ~89.1% |
| Final training loss | ~0.18 |
| Lowest validation loss | ~0.316 |

| Model | Test Accuracy |
|---|---|
| Baseline MLP | 87.3% |
| Optimized MLP | 89.4% (+2.1 points) |

### 5.6 Visualizations

Class distribution, sample images, training/validation accuracy and loss vs. epoch, confusion matrix, hyperparameter search results, baseline vs. optimized accuracy.

### 5.7 Key Observations

- Training accuracy rose from about 82% to 93%, with validation accuracy stabilizing near 89%.
- Validation loss began rising slightly in later epochs, indicating mild overfitting.
- Trouser, Sandal, Sneaker, Bag and Ankle boot are classified very well.
- Most errors occur among visually similar upper-body garments: Shirt, Pullover, Coat and T-shirt/top.
- Systematic tuning gave about +2.1 points over the baseline.

---

## 6. Experiment 3 – CNNs for Image Classification

### 6.1 Objective

To understand how CNNs work by implementing:

- The convolution operation and output-dimension calculations
- Feature map visualization
- Max pooling and average pooling
- CNN parameter calculation
- A complete CNN pipeline for CIFAR-10 and evaluation with standard metrics

### 6.2 Dataset

| Property | Value |
|---|---|
| Dataset | CIFAR-10 |
| Training / testing images | 50,000 / 10,000 |
| Classes | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Image size | 32 × 32 × 3 (RGB) |
| Per-class training images | 5,000 (perfectly balanced) |
| Preprocessing | Pixel values normalized to [0, 1] |

### 6.3 Notebook Structure

| Section | Description |
|---|---|
| 0 – Setup | TensorFlow/Keras, NumPy, Matplotlib, seaborn, scikit-learn; seeds |
| Task 1 – Load CIFAR-10 | Shapes, 10 sample images, class distribution |
| Task 2 – Convolution layer | Kernel sizes 3×3, 5×5, 7×7 and resulting feature map sizes |
| Task 3 – Hyperparameter study | Stride (1 vs. 2) and padding (same vs. valid) output dimensions |
| Task 4 – Feature map visualization | 8 feature maps after the first convolution layer |
| Task 5 – Pooling comparison | Quick 5-epoch max vs. average pooling comparison |
| Task 6 – CNN construction and training | `Input → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Dense → Softmax`; Adam, 20 epochs, batch 32 |
| Task 7 – Evaluation | Accuracy, precision, recall, F1, confusion matrix, classification report |
| Mandatory plots | All 8 required plots |
| Discussion | 6 conceptual questions (convolution vs. FC, stride, padding, pooling, feature maps, parameter efficiency) |
| Export utility | Converts `.eps` plots to `.png` and zips for submission |

### 6.4 Results

| Metric | Value |
|---|---|
| Training accuracy | 89.48% |
| Testing accuracy | 66.14% |
| Precision / Recall / F1 (weighted) | 0.6585 / 0.6614 / 0.6565 |
| Trainable parameters | 545,098 |
| Training time | 1632.28 s (~27.2 min) |

| Pooling Type (5-epoch test) | Test Accuracy |
|---|---|
| Max pooling | ~0.64 |
| Average pooling | ~0.60 |

Max pooling wins because it keeps the strongest local activation (sharper edge and texture signals) instead of diluting it by averaging.

### 6.5 Per-Class Performance (correct out of ~1000)

| Class | Correct | Notes |
|---|---|---|
| Automobile | 815 | Best performing class |
| Ship | 814 | Very well classified |
| Horse | 779 | Strong |
| Truck | 765 | Strong |
| Frog | 744 | Good |
| Deer | 633 | Moderate |
| Airplane | 596 | Some confusion with ship |
| Dog | 553 | Frequently confused with cat |
| Bird | 532 | Frequently confused with deer |
| Cat | 383 | Weakest; confused with dog and deer |

Most common confusions: cat ↔ dog (163 dogs → cat, 141 cats → dog) and bird ↔ deer (137 birds → deer). Rigid, geometrically distinct vehicles are classified far more reliably than texture-heavy, pose-variable animals.

### 6.6 Training Behaviour

| Curve | Behaviour |
|---|---|
| Training accuracy | ~0.51 → ~0.96 over 20 epochs |
| Validation accuracy | Reaches ~0.69–0.70 by epoch 2–3, then plateaus/oscillates (~0.67–0.70) |
| Training loss | Falls smoothly from ~1.38 to ~0.11 |
| Validation loss | Minimum ~0.90 around epoch 2–4, then rises to ~2.36 by epoch 19 |

The widening train/validation gap after epoch ~4 is a clear overfitting signature.

### 6.7 Mandatory Plots

Sample images, class distribution, training accuracy, validation accuracy, training loss, validation loss, feature maps after the first convolution, confusion matrix.

### 6.8 Conclusion and Recommended Improvements

The experiment verified convolution, output-dimension formulas, pooling and parameter counting, but the ~23-point train/test gap shows substantial overfitting for this shallow network. Recommended improvements: early stopping around epoch 2–4, dropout/L2, data augmentation, batch normalization, and more filters (e.g. 16 → 64) combined with the above.

---

## 7. Experiment 4 – Transfer Learning with Deep CNN Architectures

### 7.1 Objective

- Load an ImageNet-pretrained CNN (VGG16) and adapt it to a new dataset
- Freeze the convolutional base and train only new classification layers
- Fine-tune deeper layers at a low learning rate
- Run a hyperparameter study (learning rate, batch size, optimizer, dense width, frozen vs. partially unfrozen base)
- Evaluate with standard metrics and compare with classical architectures (LeNet-5, AlexNet, GoogleNet, ResNet50)

### 7.2 Dataset

| Property | Value |
|---|---|
| Dataset | CIFAR-10 (50,000 train / 10,000 test, 10 classes) |
| Native / fed size | 32 × 32 × 3, resized to 64 × 64 × 3 by a `Resizing` layer |
| Normalization | Pixels in [0, 1] |
| Balance | 5,000 training images per class |

### 7.3 Notebook Structure

| Section | Description |
|---|---|
| 0 – Setup | Imports and seeds |
| Task 1 – Dataset preparation | Load, normalize, 10 samples, dimensions |
| Task 2 – Model construction | VGG16 (`include_top=False`), frozen base → GAP → Dense(ReLU) → Dense(Softmax) |
| Task 3 – Training | Adam, lr 0.001, batch 32, 15 epochs, categorical cross-entropy |
| Task 4 – Fine-tuning | Unfreeze `block5_*`, 8 epochs at lr 1e-5, compare before/after |
| Mandatory plots | Combined accuracy/loss across both phases |
| Task 5 – Evaluation | Accuracy, precision, recall, F1, confusion matrix, classification report, misclassified samples |
| Hyperparameter study | 7 configurations × 5 epochs, ranked by validation accuracy |
| Results | Performance table and architecture comparison table |
| Additional Exercise 2 | ResNet50 backbone through the same pipeline |
| Discussion | 10 conceptual questions (AlexNet, VGG16 filter choice, Inception, residual learning, LeNet vs. ResNet, transfer learning, fine-tuning, dilated vs. transpose convolution, faster convergence, complexity) |

### 7.4 Results

| Metric (VGG16, frozen + fine-tuned) | Value |
|---|---|
| Training accuracy (final epoch) | 98.86% |
| Testing accuracy | 77.73% |
| Precision / Recall / F1 (macro) | 0.7805 / 0.7773 / 0.7757 |
| Total parameters | 14,781,642 |
| Trainable parameters (frozen phase) | 66,954 |
| Training time | 1086.6 s (580.5 s frozen + 506.1 s fine-tuning) |

| Stage | Test Accuracy |
|---|---|
| Frozen base only (15 epochs) | 65.90% |
| After fine-tuning last conv block (8 epochs) | 77.73% |
| Improvement | +11.83 points |

### 7.5 Hyperparameter Study (5 epochs)

| Configuration | Validation Accuracy | Time (s) |
|---|---|---|
| Partial frozen base (last block unfrozen) | 0.7967 | 313.8 |
| Dense units = 256 | 0.6544 | 198.6 |
| Batch size = 16, Adam | 0.6537 | 227.1 |
| Batch size = 64, Adam | 0.6533 | 191.9 |
| LR = 0.001, Adam (baseline) | 0.6501 | 196.1 |
| LR = 0.0001, Adam | 0.6172 | 197.1 |
| SGD optimizer | 0.5919 | 196.2 |

Partial unfreezing outperformed every frozen configuration within the same budget, and Adam beat SGD.

### 7.6 Per-Class Performance (correct out of 1000)

| Class | Correct | Notes |
|---|---|---|
| Ship | 934 | Best |
| Automobile | 848 | Very good |
| Truck | 836 | Strong |
| Airplane | 819 | Strong |
| Frog | 814 | Good |
| Horse | 805 | Good |
| Dog | 772 | Moderate, inflated by cat → dog errors |
| Bird | 738 | Moderate |
| Deer | 711 | Moderate |
| Cat | 496 | Weakest |

Main confusions: cat → dog (249, the largest off-diagonal cell), dog → cat (87), airplane → ship (82), horse → dog (79), deer → bird (64), truck → automobile (55).

### 7.7 Training Behaviour

| Curve | Behaviour |
|---|---|
| Training accuracy | ~0.56 → ~0.75 in the frozen phase, then jumps to ~0.99 during fine-tuning |
| Validation accuracy | ~0.66 in the frozen phase, rises to ~0.78, plateaus near epoch 6–7 of fine-tuning |
| Training loss | Falls steadily, dropping to ~0.06 by the final epoch |
| Validation loss | Falls, dips early in fine-tuning (epoch 2–3), then creeps up slightly |

The frozen phase showed little overfitting (only ~67K trainable parameters); after unfreezing, training raced ahead (98.86% vs. 77.73%) and validation loss rose, signalling mild overfitting.

### 7.8 Architecture Comparison

| Model | Parameters | Accuracy (%) | Training Time (s) |
|---|---|---|---|
| LeNet-5 | 60K | – (literature reference) | – |
| AlexNet | 61M | – (literature reference) | – |
| VGG16 (this run) | 14,781,642 | 77.73 | 1086.6 |
| GoogleNet | 6.8M | – (literature reference) | – |
| ResNet50 (this run) | 23,851,274 | 41.28 | 390.3 |

ResNet50 was run within the same short budget but, unlike VGG16, without a fine-tuning phase. Its BatchNorm statistics and deeper residual structure typically need more epochs, a higher learning rate or partial fine-tuning to adapt to small, low-resolution data.

### 7.9 Conclusion

Fine-tuning gave an 11.83-point gain, and compared with Experiment 3 (66.14% from scratch) transfer learning improved test accuracy by roughly 11–12 points. Future improvements: early stopping in the fine-tuning phase, progressive unfreezing, augmentation, longer hyperparameter sweeps, and measured (not literature) results for more backbones such as EfficientNet or MobileNet.

---

## 8. Experiment 5 – CNN Training, Regularization, Optimization and Cross-Validation

### 8.1 Objective

An end-to-end empirical study of the factors governing CNN training, using MobileNetV2 transfer learning on Oxford-IIIT Pet:

- Weight initialization comparison (Zero, Random, Xavier/Glorot, He)
- Regularization and overfitting analysis (none, L2, Dropout, BatchNorm)
- From-scratch verification of the BatchNorm formula
- Optimizer comparison (SGD, Momentum, RMSProp, Adam)
- One-variable-at-a-time hyperparameter tuning
- Feature extraction vs. fine-tuning
- 5-fold cross-validation reporting mean and standard deviation
- Final evaluation on an untouched test set
- An additional exercise proposing and cross-validating two new configurations

### 8.2 Dataset

| Property | Value |
|---|---|
| Dataset | Oxford-IIIT Pet (via `tensorflow_datasets`) |
| Classes | 37 pet breeds (cats and dogs) |
| Splits | `train` split divided 80/20 into train/validation; `test` split untouched until the final evaluation |
| Image size | Resized to 224 × 224 × 3 |
| Preprocessing | `mobilenet_v2.preprocess_input` (pixels to [-1, 1]) |
| Augmentation (where used) | Random horizontal flip, random brightness |

### 8.3 Notebook Structure

| Section | Description |
|---|---|
| 0 – Setup | Imports, seeds, plot style |
| 3 – Dataset & setup | `tfds` loading and `tf.data` pipelines |
| 4 – Model builder | Single `build_model()` parameterizing initializer, dropout, BatchNorm, L2, frozen/fine-tuned base, `fine_tune_at`, optimizer and learning rate |
| 5 – Weight initialization | Zero, Random, Xavier, He on the classifier head (Plots 1–2) |
| 6 – Regularization | None, L2, Dropout, BatchNorm (Plots 3–4) |
| 7 – Batch Normalization | Worked example for x = [2, 4, 6, 8], checked against Keras; with/without BN (Plot 5) |
| 8 – Optimizers | SGD, Momentum, RMSProp, Adam with comparison table (Plots 6–7) |
| 9 – Hyperparameters | LR {0.001, 0.0001}, batch {16, 32, 64}, dropout {0.0, 0.25, 0.5} (Plots 8–10) |
| 10 – Transfer learning | Case A frozen base vs. Case B fine-tuning at `fine_tune_at = 100`, lr 1e-5 (Plots 11–12) |
| 11 – K-fold CV | 5-fold CV over candidate configurations C1–C4, mean ± SD (Plot 13) |
| 12 – Final evaluation | Best configuration retrained on train+val, evaluated once on test; confusion matrix (Plot 14), misclassified images (Plot 15) |
| 13 – Overall results | Best result tracked at each stage |
| 14 – Required inference | 2–3 line inference for every plot |
| 15 – Discussion | 23 conceptual questions |
| 16 – Additional exercise | Aggressive fine-tune (`fine_tune_at = 50`, low dropout) and frozen-base high-dropout RMSProp configuration |
| 17 – Expected outcome | Summary of how each stage shapes final performance |

### 8.4 Methodology Highlights

| Topic | Detail |
|---|---|
| Reusable builder | Every section changes only the variable under study, following the manual's one-hyperparameter-at-a-time rule |
| Initialization | Compared on the head with a frozen base to isolate effects on convergence |
| Regularization | Training/validation gap compared across four strategies; BN worked example reproduced by hand |
| Optimizers | Identical architecture and learning rate; table of final loss, best validation accuracy, epochs to converge, wall-clock time |
| Transfer learning | Feature extraction vs. fine-tuning with a much smaller learning rate (1e-5) |
| Model selection | CV on the training pool only; best mean-accuracy configuration retrained and tested once |
| Final evaluation | Accuracy, precision, recall, F1, full 37×37 confusion matrix, top confused breed pairs |

### 8.5 Mandatory Plots

1. Training loss, weight initialization 2. Validation accuracy, initialization 3. Accuracy, regularization 4. Loss, regularization 5. With vs. without BatchNorm 6. Training loss, optimizers 7. Validation accuracy, optimizers 8. Learning rate vs. validation accuracy 9. Batch size vs. validation accuracy 10. Dropout rate vs. validation accuracy 11. Feature extraction vs. fine-tuning 12. Loss before vs. after fine-tuning 13. 5-fold CV accuracy ± SD 14. Confusion matrix (37 breeds) 15. Misclassified images 16. Additional-exercise comparison. Figures are saved as PNG and 600 dpi EPS.

### 8.6 Discussion Question Coverage (23)

| Topic | Questions |
|---|---|
| Parameters vs. hyperparameters, initialization | Q1–Q4 |
| Overfitting, Dropout, BatchNorm | Q5–Q9 |
| Optimizers, learning rate, batch size | Q10–Q13 |
| Stride, padding, MobileNetV2 efficiency | Q14–Q16 |
| Transfer learning | Q17–Q19 |
| Cross-validation and model selection | Q20–Q23 |

### 8.6 Results Status

The notebook was supplied as a fully wired but **unexecuted template**: inference cells contain `[...]` placeholders. After a full run, populate: best initializer, best regularization, best optimizer, best LR/batch/dropout, feature-extraction vs. fine-tuned accuracy, winning CV configuration (mean ± SD), and final test metrics.

---

## 9. Experiment 6 – RNN, LSTM and GRU

### 9.1 Objective

- Backpropagation Through Time and vanishing/exploding gradients
- LSTM gates (forget, input, output) and GRU gates (reset, update)
- Human Activity Recognition on UCI HAR
- CNN–RNN pipelines for video understanding
- Encoder–decoder sequence-to-sequence learning
- Comparative evaluation with standard classification metrics

### 9.2 Datasets

| Dataset | Details |
|---|---|
| UCI HAR (primary) | 7,352 train / 2,947 test windows; 6 classes; 128 time steps × 9 channels; activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING |
| Subset used | Stratified 2,000 windows (~333 per class), split 70/15/15; `StandardScaler` fitted on training data only |
| Simulated video | Synthetic RGB frames (224 × 224 × 3) per class; frozen MobileNetV2 extracts features; 10 frames per clip give a `(10, feat_dim)` sequence |
| Seq2Seq | Integer sequences of length 4 mapped to their sorted version; second exercise uses input length 6, output length 4 |

### 9.3 Notebook Structure

| Section | Description |
|---|---|
| 0 – Setup | Imports and seeds |
| 1 – Data | Download, raw inertial signals, subset, split, normalize |
| 2 – Temporal visualization (Plot 1) | Sensor signals: 3 activities × 3 channels |
| 3 – BPTT exercise | 3-step Vanilla RNN forward pass verified against Keras `SimpleRNNCell`; vanishing gradient illustration |
| 4 – Model definitions | Unified `build_model(rnn_type)` for RNN/LSTM/GRU |
| 5 – Training (Plots 2–3) | 30 epochs, early stopping, ReduceLROnPlateau |
| 6 – Evaluation (Plots 4–5) | Accuracy, precision, recall, F1; confusion matrices; metric/parameter bar chart |
| 7 – Sequence length study (Plot 6) | Lengths 32/64/128 vs. macro F1 and cost |
| 8 – CNN–LSTM video (Plots 7–9) | MobileNetV2 features → LSTM, curves and confusion matrix |
| 9 – Seq2Seq (same length) | Teacher forcing; token and sequence accuracy |
| 10 – Seq2Seq (different length) | IN_LEN = 6, OUT_LEN = 4 |
| 11 – Consolidated results | RNN, LSTM, GRU, CNN–LSTM and Seq2Seq |
| 12 – Discussion | 25 conceptual questions |
| 13 – Additional exercises (7) | Unit count; GRU vs. LSTM; stacked LSTM; bidirectional LSTM; sequence length vs. cost; CNN-GRU vs. CNN-LSTM; variable-length Seq2Seq |

### 9.4 Training Configuration

| Setting | Value |
|---|---|
| Epochs | 30 |
| EarlyStopping | patience 5 on `val_loss` |
| ReduceLROnPlateau | factor 0.5, patience 3 |
| Application | Applied uniformly to all three architectures |

### 9.5 Expected Behaviour and Parameter Counts (32 units)

| Architecture | Approx. Trainable Parameters |
|---|---|
| Vanilla RNN | ~3,000 |
| LSTM | ~10,000 |
| GRU | ~8,000 |

| Curve | Expected Behaviour |
|---|---|
| RNN training loss | Decreases slowly, may plateau (vanishing gradients) |
| RNN validation loss | Plateaus early, largest train–val gap |
| LSTM/GRU training loss | Decreases smoothly (gating mitigates vanishing gradient) |
| LSTM/GRU validation loss | Tracks training more closely |

### 9.6 Additional Exercise Findings (expected/typical)

| Exercise | Finding |
|---|---|
| 1 – Unit count (16/32/64) | Gain 16 → 32, diminishing returns at 64 with higher cost |
| 2 – GRU vs. LSTM | GRU has ~25% fewer parameters; F1 gap typically < 2 points |
| 3 – Stacked 2-layer LSTM | Marginal gain (~1–2%), training time doubles |
| 4 – Bidirectional LSTM | Doubles parameters; suited to offline tasks, impractical for real-time |
| 5 – Sequence length vs. cost | Parameters unchanged, epoch time grows linearly with T, F1 saturates |
| 6 – CNN-LSTM vs. CNN-GRU | Similar F1 on 10-frame clips, GRU faster |
| 7 – Variable-length Seq2Seq | Encoder–decoder handles IN_LEN ≠ OUT_LEN with no architectural change |

### 9.7 Mandatory Plots

| # | File | Description |
|---|---|---|
| 1 | `plot1_sensor_signals.png` | Sensor signal vs. time step |
| 2 & 3 | `plot2_3_loss_accuracy.png` | Loss/accuracy for RNN, LSTM, GRU |
| 4 | `plot4_confusion_matrices.png` | Test confusion matrices |
| 5 | `plot5_comparison_bar.png` | Accuracy, F1, parameters, training time |
| 6 | `plot6_seqlen_f1.png` | Sequence length vs. macro F1 |
| 7 | `plot7_video_frames.png` | Synthetic video frames |
| 8 | `plot8_video_training.png` | CNN–LSTM training curves |
| 9 | `plot9_video_confusion.png` | CNN–LSTM confusion matrix |

### 9.8 Conceptual Highlights

| Concept | Summary |
|---|---|
| Vanishing gradient | Gradient scales as `|Wh · σ'|^k` over k steps; with Wh = 0.8 it is ≈ 0 within ~20 steps |
| LSTM gating | Cell state flows with linear interactions, protected by forget (erase), input (write), output (expose) gates |
| GRU | Merges forget and input into one update gate and drops the separate cell state, ~25% fewer parameters |
| CNN–RNN | The CNN supplies "what" (spatial content), the RNN supplies "how it changes" |
| Seq2Seq | Encoder compresses the input to a context vector; decoder generates token by token. Teacher forcing during training, own outputs at inference |
| Sequence vs. token accuracy | Sequence accuracy ≤ token accuracy, since one wrong token invalidates the whole sequence |

### 9.9 Results Status

The source README leaves its result tables unpopulated, so no measured accuracy, F1 or Seq2Seq values are reported here. Fill in the HAR table (RNN, LSTM, GRU, CNN–LSTM) and Seq2Seq metrics (token accuracy, sequence accuracy, final train/val loss) after execution. The expected conclusion is that LSTM and GRU clearly outperform Vanilla RNN, that GRU matches LSTM at lower cost, and that early stopping and dropout are essential on small subsets.

---

## 10. Experiment 7 – Autoencoders

### 10.1 Objective

- Encoder–latent–decoder structure of a fully connected autoencoder
- Convolutional autoencoders for spatially aware reconstruction
- Denoising autoencoders trained against clean targets (Gaussian and salt-and-pepper corruption)
- Variational autoencoders: probabilistic latents and the reparameterization trick
- Quantitative evaluation with MSE, MAE and SSIM
- Latent-space visualization, interpolation and sampling

### 10.2 Dataset

| Property | Value |
|---|---|
| Source | Keras/TensorFlow `mnist` |
| Image size | 28 × 28 × 1 grayscale |
| Training / test subset | 10,000 / 2,000 images |
| Classes | 10 (labels not used as targets; the image itself is the target) |
| Normalization | [0, 255] → [0, 1] |

| Noise Type | Formulation | Levels |
|---|---|---|
| Gaussian | x̃ = x + n, n ~ N(0, σ²), clipped to [0, 1] | σ ∈ {0.1, 0.2, 0.3} |
| Salt-and-pepper | Random pixels set to 0 or 1 | p ∈ {0.05, 0.10, 0.20} |

### 10.3 Notebook Structure

| Section | Description |
|---|---|
| 0 – Setup | TensorFlow-Keras, NumPy, Matplotlib, scikit-image; load and normalize MNIST |
| 1 – FC-AE (Plots 1–2) | `784 → 128 → 32 → 16 → 32 → 128 → 784`; Adam + BCE |
| 2 – CAE (Plot 3) | Conv(32)→MaxPool→Conv(64)→MaxPool→Conv(64) encoder; UpSampling+Conv decoder |
| 3 – Denoising CAE (Plots 4–5) | Corrupted input, clean target; quality across noise levels |
| 4 – VAE (Plots 6–8) | 2-D latent, reparameterization; latent scatter, sampling, interpolation |
| 5 – Model comparison | MSE/MAE/SSIM/parameter/time table |
| 6 – Error distribution (Plot 9) | Per-image MSE histogram; five hardest images |
| 7 – Latent-dimension study (Plot 10) | Latent sizes 4/8/16/32 |
| 8 – Discussion | 25 conceptual questions |
| 9 – Additional exercises (7) | Listed in §10.7 |

### 10.4 Results

| Model | MSE | MAE | SSIM | Parameters | Time (s) |
|---|---|---|---|---|---|
| Fully Connected AE | 0.022507 | 0.058655 | 0.735063 | 211,040 | 18.79 |
| Convolutional AE | 0.002974 | 0.016280 | 0.968968 | 74,497 | 19.48 |
| Denoising CAE | 0.005036 | 0.022221 | 0.940905 | 74,497 | 16.57 |
| VAE | 0.044173 | 0.102887 | 0.483118 | 284,933 | 29.16 |

| VAE Loss Component | Value |
|---|---|
| Reconstruction loss | 149.843140 |
| KL loss | 6.049046 |
| Total loss | 155.892181 |

| Gaussian σ | MSE | MAE | SSIM |
|---|---|---|---|
| 0.1 | 0.004042 | 0.018699 | 0.955667 |
| 0.2 | 0.005047 | 0.022242 | 0.940823 |
| 0.3 | 0.007544 | 0.030127 | 0.873935 |

### 10.5 Key Training Observations

| Model | Behaviour |
|---|---|
| FC-AE | Loss falls steeply for ~5 epochs then flattens near 0.125; curves aligned, no strong overfitting |
| CAE vs. FC-AE | Lower MSE/MAE, higher SSIM, ~3× fewer parameters (locality and weight sharing) |
| Denoising CAE | Recovers digit structure under visible corruption; degrades gradually with noise |
| VAE | Blurrier than the CAE, but continuous latent space supports sampling and interpolation |

### 10.6 Mandatory Plots

Original vs. reconstructed (FC-AE); FC-AE train/validation loss; FC-AE vs. CAE; clean vs. noisy vs. denoised; noise level vs. MSE/MAE/SSIM; VAE 2-D latent scatter by digit; random VAE samples; latent interpolation; per-image error distribution; latent dimension vs. MSE.

### 10.7 Additional Exercises

| Exercise | Finding |
|---|---|
| 1 – Latent-channel sweep (4/8/16/32, CAE) | MSE falls 0.019347 → 0.004461; diminishing returns beyond 16 |
| 2 – Gaussian vs. salt-and-pepper | Salt-and-pepper hurts more at matched levels (SSIM 0.770 at p = 0.20 vs. 0.874 at σ = 0.30) |
| 3 – DAE training noise | σ = 0.1 gives SSIM 0.9557; σ = 0.3 gives 0.9020, trading clean fidelity for robustness |
| 4 – Conv2DTranspose vs. UpSampling2D | Transposed convolution far better (MSE 0.000835 vs. 0.002974; SSIM 0.9902 vs. 0.9690) |
| 5 – VAE latent 2 vs. 8 | MSE 0.046 → 0.024; SSIM 0.454 → 0.734, but no longer directly visualizable |
| 6 – β-VAE (0.5 → 5.0) | Not monotonic; SSIM peaks near β = 1.0 (0.4378) |
| 7 – 100-sample grid, centroid interpolation | 10×10 grid shows more diversity and the same class overlap; centroid-to-centroid (1 → 7) is smoother than arbitrary-sample interpolation |

### 10.8 Conceptual Highlights

| Concept | Summary |
|---|---|
| Autoencoder | `z = f_θ(x)`, `x̂ = g_φ(z)`; minimizes `(1/N)Σ‖x_i − x̂_i‖²`; the bottleneck prevents a trivial identity mapping |
| FC vs. conv | Flattening discards spatial structure; convolution shares weights to learn stroke-like filters |
| Denoising | Corrupted input, clean target; the bottleneck acts as a structural filter |
| VAE | `q_φ(z|x) = N(μ, diag σ²)`, `z = μ + σ ⊙ ε`; loss = reconstruction + `D_KL(q‖p)`, giving a smooth, sampleable latent space |

### 10.9 Conclusion

Deterministic autoencoders (FC-AE, CAE, DAE) optimize pure reconstruction, while the VAE trades fidelity for a structured generative latent space. The right choice depends on whether the goal is faithful reconstruction or new-sample generation.

---

## 11. Experiments 8 & 9 – Attention in a Pretrained ResNet50

### 11.1 Objective

To understand how attention mechanisms modify a pretrained CNN and how insertion position influences classification. A single ImageNet-pretrained ResNet50 is used throughout. The study covers channel, spatial and sequential channel–spatial attention, lightweight self-attention, attention at P1–P4 (single and multiple stages), attention order, spatial kernel size, additional mechanisms (SE, ECA, CBAM), and qualitative analysis with internal attention maps and Grad-CAM.

### 11.2 Dataset and Positions

| Property | Value |
|---|---|
| Dataset | TensorFlow Flowers (`tf_flowers`) |
| Classes | daisy, dandelion, roses, sunflowers, tulips |
| Image size | 224 × 224 × 3, ResNet50 `preprocess_input` |
| Split | 70 : 15 : 15 → 2,569 / 550 / 551 images, identical for every run |
| Test class counts | 126 dandelion, 98 daisy, 118 tulips, 111 sunflowers, 98 roses |
| Augmentation | Flip, small rotation, zoom, translation, mild contrast |

| Position | Location | Resolution | Channels |
|---|---|---|---|
| P1 | After conv2_x | 56 × 56 | 256 |
| P2 | After conv3_x | 28 × 28 | 512 |
| P3 | After conv4_x | 14 × 14 | 1024 |
| P4 | After conv5_x | 7 × 7 | 2048 |

All attention modules preserve the tensor shape.

### 11.3 Training Protocol

| Setting | Value |
|---|---|
| Classifier | GAP → Dropout(0.3) → Dense(5) |
| Optimizer / loss | Adam / categorical cross-entropy |
| Stage 1 | LR 1e-3, 3 epochs, backbone frozen |
| Stage 2 | LR 1e-4, 2 epochs, conv5_x unfrozen |
| Batch / seed | 16 / 42 |
| Callbacks | Early stopping on val accuracy (patience 5, best weights restored) |
| Inference time | Mean of 50 single-image forward passes |
| Mode | QUICK RUN (5 epochs total), single run per model |

### 11.4 Notebook Structure

| Section | Description |
|---|---|
| 0 – Setup | Imports, seeds, hyperparameters, QUICK RUN switch |
| 1 – Dataset | Fixed split, sample image per class |
| 2 – Backbone | Stage extraction conv2_x–conv5_x |
| 3 – Attention modules | Channel (SE-style), spatial 7×7, Channel→Spatial and Spatial→Channel, self-attention, ECA, CBAM |
| 4 – Model builder | Any mechanism at any stage or stage combination |
| 5 – Training utilities | Two-stage training, metrics, timing, checkpoint/resume |
| 6 – Baseline | Pretrained ResNet50 without attention |
| 7–11 – Experiments 1–5 | Type, position, multiple positions, order, kernel size |
| 12 – Additional task | SE, ECA, CBAM |
| 13 – Main result table | Consolidated results |
| 14 – Graphical analysis | Accuracy/loss curves, accuracy and F1 bars, accuracy vs. parameters, F1 vs. inference time |
| 15 – Confusion matrices | Baseline vs. Channel+Spatial (P3) |
| 16 – Grad-CAM | Baseline and attention model |
| 17 – Internal attention maps | Channel weights, spatial heatmap/overlay, self-attention response, before/after feature maps |
| 18 – Position visualization | Spatial attention at P2, P3, P4 |
| 19 – Qualitative comparison | Five test images |
| 20 – Discussion | 35 manual questions |
| 21 – Save results | Export metrics and figures |

### 11.5 Main Result Table

| Model | Position | Accuracy | Precision | Recall | Macro F1 | Trainable Params | Total Params | Time (ms) |
|---|---|---|---|---|---|---|---|---|
| A0 Baseline | – | 0.9328 | 0.9358 | 0.9305 | 0.9323 | 14,986,245 | 23,597,957 | 95.46 |
| A1 Channel | P3 | 0.9365 | 0.9363 | 0.9377 | 0.9359 | 15,118,405 | 23,730,117 | 90.41 |
| A2 Spatial 7×7 | P3 | 0.9256 | 0.9304 | 0.9223 | 0.9248 | 14,986,344 | 23,598,056 | 109.36 |
| A3 Channel–Spatial | P3 | 0.9328 | 0.9341 | 0.9312 | 0.9316 | 15,118,504 | 23,730,216 | 96.77 |
| A4 Self-Attention | P3 | **0.9437** | 0.9428 | 0.9446 | **0.9435** | 16,298,246 | 24,909,958 | 115.30 |
| A5 Combined | P3 | 0.9347 | 0.9375 | 0.9319 | 0.9340 | 16,430,505 | 25,042,217 | 97.22 |
| B1 Channel–Spatial | P1 | 0.9093 | 0.9102 | 0.9095 | 0.9085 | 14,994,808 | 23,606,520 | 106.76 |
| B2 Channel–Spatial | P2 | 0.9147 | 0.9187 | 0.9144 | 0.9140 | 15,019,656 | 23,631,368 | 104.28 |
| B4 Channel–Spatial | P4 | 0.9183 | 0.9202 | 0.9147 | 0.9166 | 15,512,808 | 24,124,520 | 105.93 |
| C Channel–Spatial | P2+P3+P4 | 0.9256 | 0.9266 | 0.9270 | 0.9249 | 15,678,478 | 24,290,190 | 94.10 |
| C Channel–Spatial | P3+P4 | 0.9020 | 0.9062 | 0.9032 | 0.9013 | 15,645,067 | 24,256,779 | 97.22 |
| D Spatial→Channel | P3 | 0.9310 | 0.9333 | 0.9282 | 0.9299 | 15,118,504 | 23,730,216 | 114.68 |
| ECA | P3 | 0.9111 | 0.9105 | 0.9117 | 0.9106 | 14,986,250 | 23,597,962 | 103.11 |
| CBAM | P3 | 0.9074 | 0.9064 | 0.9060 | 0.9059 | 15,118,504 | 23,730,216 | 106.64 |
| Spatial 3×3 | P3 | 0.9165 | 0.9210 | 0.9131 | 0.9157 | 14,986,264 | 23,597,976 | 91.20 |
| Spatial 5×5 | P3 | 0.9129 | 0.9135 | 0.9114 | 0.9119 | 14,986,296 | 23,598,008 | 103.93 |

### 11.6 Experiment-by-Experiment Findings

| Experiment | Result |
|---|---|
| 1 – Attention type (P3) | Self-attention best (+1.09 points), channel next (+0.36); spatial, ECA, CBAM below baseline |
| 2 – Position (Channel–Spatial) | P3 best (0.9328), then P4 (0.9183), P2 (0.9147), P1 (0.9093) |
| 3 – Multiple positions | P3 + P4 was 3.1 points below P3 alone; P2+P3+P4 0.72 points below; extra parameters ~0.53–0.56 M |
| 4 – Order | Channel→Spatial 0.9328 vs. Spatial→Channel 0.9310: a single test image apart, so no conclusion on order |
| 5 – Kernel size | 3×3 0.9165, 5×5 0.9129, 7×7 0.9256; no steady trend; spatial module adds 19 / 51 / 99 parameters |
| Additional – SE, ECA, CBAM | ECA adds 5 parameters and CBAM ~132 k; both end below the baseline |

### 11.7 Added Parameters per Mechanism

| Mechanism | Added Parameters |
|---|---|
| ECA | 5 |
| Spatial (7×7) | 99 |
| Channel / SE | 132,160 |
| CBAM | 132,259 |
| Self-Attention | 1,312,001 |

### 11.8 Training Behaviour (Final Epoch)

| Model | Train Acc | Val Acc | Train Loss | Val Loss |
|---|---|---|---|---|
| A0 Baseline | 0.9677 | 0.9364 | 0.0970 | 0.2210 |
| A1 Channel | 0.9611 | 0.9109 | 0.1190 | 0.2784 |
| A2 Spatial 7×7 | 0.9611 | 0.9200 | 0.1075 | 0.2406 |
| A3 Channel–Spatial | 0.9642 | 0.9182 | 0.1174 | 0.2349 |
| A4 Self-Attention | 0.9681 | 0.9218 | 0.0966 | 0.2584 |
| A5 Combined | 0.9560 | 0.9218 | 0.1318 | 0.2381 |
| D Spatial→Channel | 0.9580 | 0.9345 | 0.1203 | 0.2269 |

- Self-attention started faster (first-epoch training accuracy 0.771 vs. 0.745; validation 0.902 vs. 0.884) but all models converged to about 0.965–0.968 training accuracy.
- All curves dip when fine-tuning starts. Channel+Spatial was the least stable (validation loss rising to 0.42 vs. 0.307 for the baseline).
- Final train–validation gaps: baseline 0.031, self-attention 0.046, Channel+Spatial 0.046, so attention did not reduce the gap.
- No attention model finished with a lower validation loss than the baseline (0.221).

### 11.9 Statistical Reliability

| Point | Detail |
|---|---|
| Test-set size | 551 images; one image = 0.18 points; error about ±2 points |
| McNemar, self-attention vs. baseline | Right on 18 images where the baseline is wrong, wrong on 12 where it is right (p = 0.36) |
| McNemar, Channel+Spatial vs. baseline | 14 images each way (p = 1.0) |
| Runs | One run per model; seed-repeated runs are needed to confirm rankings |

### 11.10 Confusion Matrix Analysis (Baseline vs. Channel+Spatial P3; 514/551 correct for both)

| Class | Notes |
|---|---|
| Tulips | Best class (recall 0.966 → 0.983); F1 0.912 → 0.943 |
| Sunflowers | F1 0.932 → 0.942 |
| Daisy | Recall 0.908 → 0.939 |
| Dandelion | Got worse: dandelion → daisy errors went from 1 to 6; F1 0.964 → 0.939 |
| Roses | Recall 0.898 → 0.867; roses → tulips is the most common confusion (7 → 9 images) |

Overall accuracy is unchanged, but errors move between classes, so the benefit is not uniform.

### 11.11 Qualitative Analysis

| Visualization | Finding |
|---|---|
| Baseline Grad-CAM (5 images) | Seed head for dandelion; sunflowers and nearby sky for the sunflower image; stem/leaf, not petals, for tulips; spread-out activation for the two cluttered-field errors |
| Channel+Spatial Grad-CAM | Almost uniform with a corner spot; taken at the spatial-attention output layer, so not comparable with the baseline (conv5_block3_out) |
| Channel weights (top-20 of 1024) | All ≈ 1.0, a saturated sigmoid; no group stands out |
| Spatial maps (P2/P3/P4) | Nearly uniform; variation only at image borders, likely a zero-padding effect of the 7×7 convolution |
| Self-attention (P3) | Mean response concentrated on one location near the flower centre |
| Feature maps before vs. after | Almost identical, consistent with unchanged Channel+Spatial accuracy |
| Predictions | Channel+Spatial fixes the tulip field the baseline called roses; both call the dandelion meadow sunflowers; each model is right on 14 images the other misses |

Because only one or five images were inspected and layers differ between models, no claim of improved localization or background suppression is made.

### 11.12 Required Visualizations (Report)

Sample images; accuracy curves; loss curves; accuracy and F1 bar charts; accuracy vs. parameters; F1 vs. inference time; confusion matrices; feature maps before/after attention; channel-attention weights; spatial heatmap and overlay; self-attention response; baseline and attention Grad-CAM; attention maps at P2, P3, P4.

### 11.13 Conceptual Highlights

| Concept | Summary |
|---|---|
| Feature hierarchy | Resolution falls (56 → 7) while channels rise (256 → 2048); early = edges/textures, middle = parts/shapes, late = semantics |
| Channel attention | `M_c(F) = σ(W₂ δ(W₁ z))`, r = 16: which channels matter |
| Spatial attention | `M_s(F) = σ(Conv₇ₓ₇([F_avg ‖ F_max]))`: which locations matter |
| Sequential | `F_cs = M_s(F_c) ⊙ F_c`: what, then where |
| Self-attention | `Y = softmax(QKᵀ/√d)V`, residual `F + γ·Y`; N × N matrix grows with the square of the number of positions, so use at P3/P4 |
| Internal attention vs. Grad-CAM | Attention is computed in the forward pass; Grad-CAM is a post-hoc gradient explanation |

### 11.14 Cost–Benefit Summary

| Mechanism | Cost | Benefit |
|---|---|---|
| Channel (SE) | +0.56% parameters | +0.36 points accuracy, no clear slowdown |
| Self-attention | +5.6% parameters, ~21% slower inference | +1.09 points accuracy (not statistically significant by McNemar) |
| ECA, CBAM, Spatial | Small to moderate | No gain over baseline |

### 11.15 Conclusion

Attention effectiveness depends on type, position, feature resolution, feature semantics and model complexity together. Self-attention at P3 was best, P3 was the best position, repeating attention across stages did not help, and larger modules were not systematically better. All conclusions carry the caveats above: single runs, a 5-epoch schedule, a small test set and mismatched Grad-CAM layers.

---

## 12. Cross-Experiment Analysis

| Theme | Evidence |
|---|---|
| Convolution beats flattening | CAE vs. FC-AE (Exp 7); CNNs vs. MLP-style models (Exp 2–3) |
| Pretraining is a large lever | Exp 4: +11.83 points from fine-tuning; partial unfreezing beat every frozen setup |
| Capacity needs regularization | Exp 3 and Exp 4 show train/validation divergence |
| Complexity does not guarantee accuracy | Exp 4 (ResNet50 vs. VGG16 in a short budget), Exp 8–9 (Combined vs. self-attention; ECA/CBAM below baseline) |
| Statistical caution | Exp 8–9 reports test-set noise and McNemar tests |
| Recurring hard classes | Cat/dog (Exp 3–4); shirt/pullover/coat (Exp 2) |
| Visually similar classes drive errors | Roses vs. tulips (Exp 8–9), consistent with the above |

| Experiment | Best Single Number | Metric |
|---|---|---|
| 1 | 96.36% | Accuracy |
| 2 | 89.4% | Test accuracy |
| 3 | 66.14% | Test accuracy |
| 4 | 77.73% | Test accuracy (VGG16 fine-tuned) |
| 7 | 0.9690 | SSIM (CAE); 0.9902 with Conv2DTranspose |
| 8 & 9 | 94.37% | Test accuracy (self-attention, P3) |

## 13. Repository Layout

```
CS3807-Deep-Learning-Lab/
├── README.md                          # this file
├── Exp1_Perceptron_Banknote/
├── Exp2_MLP_FashionMNIST/
├── Exp3_CNN_CIFAR10/
├── Exp4_TransferLearning_CIFAR10/
├── Exp5_MobileNetV2_OxfordPets/
├── Exp6_RNN_LSTM_GRU/
├── Exp7_Autoencoders_MNIST/
└── Exp8_9_Attention_ResNet50/
    ├── Lab_8_code.ipynb
    ├── LAB_8_report.pdf
    └── README.md
```

Each experiment folder holds its notebook, a `figures/` directory and its own README.

## 14. Technology Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Programming language |
| TensorFlow / Keras | Model building and training |
| TensorFlow Datasets | `tf_flowers`, `oxford_iiit_pet` |
| scikit-learn | Metrics, cross-validation, randomized search, baselines |
| scikit-image | SSIM (Exp 7) |
| NumPy / pandas | Numerical work and result tables |
| Matplotlib / seaborn | Plots and confusion matrices |
| Jupyter / Google Colab (GPU) | Development and execution |

## 15. Reproducibility and Running the Experiments

1. Open the notebook in Google Colab or a local Jupyter environment.
2. For Experiments 4–9, set **Runtime → Change runtime type → GPU**.
3. Run all cells top to bottom; seeds are fixed (seed 42 where stated).
4. Experiments 8 & 9 include a QUICK RUN switch (5 epochs); disable it for the full 25-epoch schedule.
5. Figures are written to each experiment's `figures/` directory (PNG, and EPS for report submission where noted).
6. For controlled comparisons (Exp 5, 8, 9), the data split, preprocessing, augmentation, optimizer, schedule and classifier are held identical so only the factor under study changes.

## 16. Known Limitations

| Experiment | Limitation |
|---|---|
| 3 | Shallow network overfits substantially; no regularization or augmentation |
| 4 | VGG16 and ResNet50 not given identical schedules; other architectures are literature references |
| 5 | Notebook supplied unexecuted; results and inferences contain placeholders |
| 6 | Result tables unpopulated; video data is synthetic |
| 7 | Reduced MNIST subsets (10,000 / 2,000 images) |
| 8 & 9 | Single run per model, 5-epoch schedule, 551-image test set, Grad-CAM layer mismatch |

## 17. Overall Conclusion

Across nine experiments, the laboratory traces how deep learning capability is built up: from a linear boundary learned by one neuron, to spatial structure exploited through convolution, to reuse of pretrained knowledge, to control of training dynamics, to sequence modeling, learned latent representations and finally attention. The consistent lessons are that architecture should match data structure, pretrained features are a powerful starting point, added capacity must be paired with regularization and honest evaluation, and no single accuracy number tells the full story: it should be read together with complexity, per-class behaviour and statistical uncertainty.

## 18. References

1. Goodfellow, I., Bengio, Y. & Courville, A. — *Deep Learning* (MIT Press, 2016)
2. Bishop, C. M. — *Pattern Recognition and Machine Learning*
3. Haykin, S. — *Neural Networks and Learning Machines*
4. Simonyan, K. & Zisserman, A. — *Very Deep Convolutional Networks for Large-Scale Image Recognition*
5. He, K., Zhang, X., Ren, S. & Sun, J. — *Deep Residual Learning for Image Recognition* (CVPR 2016)
6. Sandler, M. et al. — *MobileNetV2: Inverted Residuals and Linear Bottlenecks*
7. Ioffe, S. & Szegedy, C. — *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*
8. Kingma, D. P. & Ba, J. — *Adam: A Method for Stochastic Optimization*
9. Hochreiter, S. & Schmidhuber, J. — *Long Short-Term Memory* (Neural Computation, 1997)
10. Cho, K. et al. — *Learning Phrase Representations using RNN Encoder–Decoder* (EMNLP 2014)
11. Howard, A. et al. — *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications* (arXiv 2017)
12. Kingma, D. P. & Welling, M. — *Auto-Encoding Variational Bayes* (ICLR 2014)
13. Vincent, P. et al. — *Stacked Denoising Autoencoders* (JMLR 2010)
14. Hu, J. et al. — *Squeeze-and-Excitation Networks* (TPAMI 2020)
15. Woo, S. et al. — *CBAM: Convolutional Block Attention Module* (ECCV 2018)
16. Wang, Q. et al. — *ECA-Net: Efficient Channel Attention for Deep CNNs* (CVPR 2020)
17. Hou, Q., Zhou, D. & Feng, J. — *Coordinate Attention for Efficient Mobile Network Design* (CVPR 2021)
18. Vaswani, A. et al. — *Attention Is All You Need* (NeurIPS 2017)
19. Selvaraju, R. R. et al. — *Grad-CAM* (ICCV 2017)
20. Datasets: UCI Banknote Authentication, Fashion-MNIST, CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html), Oxford-IIIT Pet, UCI HAR, MNIST, TensorFlow Flowers
21. TensorFlow — https://www.tensorflow.org · Keras — https://keras.io · scikit-image — https://scikit-image.org
