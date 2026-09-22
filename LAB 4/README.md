CS3807 – Deep Learning Laboratory
Experiment 4: Comparative Study of Deep CNN Architectures Using Transfer Learning
Institution: Shiv Nadar University Chennai
Degree & Branch: B.Tech Artificial Intelligence & Data Science, Semester V
Subject Code: CS3807 – Deep Learning Laboratory
Academic Year: 2026–27

1. Objective
To understand and apply transfer learning for image classification by implementing:

- Loading a CNN pretrained on ImageNet (VGG16) and adapting it to a new dataset
- Freezing a convolutional base and training only new classification layers
- Fine-tuning the deeper layers of the pretrained network at a low learning rate
- A hyperparameter study across learning rate, batch size, optimizer, dense-layer width, and frozen-vs-partial-unfrozen base
- Model evaluation using standard classification metrics
- A comparison of transfer learning against classical CNN architectures (LeNet-5, AlexNet, GoogleNet, ResNet50)

2. Dataset Description
Dataset: CIFAR-10

| Property | Value |
|---|---|
| Training images | 50,000 |
| Testing images | 10,000 |
| Number of classes | 10 |
| Image size (native) | 32 × 32 × 3 (RGB) |
| Image size (fed to backbone) | Resized to 64 × 64 × 3 |
| Classes | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Images per class (training) | 5,000 (perfectly balanced) |

Pixel values were normalized to the range [0, 1] prior to training. Since CIFAR-10 images (32×32) are smaller than the pretrained backbone's expected input, a `Resizing` layer upsamples them to 64×64 before entering the frozen convolutional base.

3. Contents of the Notebook (Experiment_4_Transfer_Learning_CIFAR10.ipynb)
The notebook is organized into the following sections, matching the lab manual's task structure:

| Section | Description |
|---|---|
| 0. Imports and Setup | Imports TensorFlow/Keras, NumPy, pandas, Matplotlib, seaborn, scikit-learn metrics; sets random seeds for reproducibility |
| Task 1 – Dataset Preparation | Loads CIFAR-10, normalizes pixel values to [0,1], displays 10 sample images, prints dataset dimensions |
| Task 2 – Transfer Learning Model Construction | Loads VGG16 pretrained on ImageNet, removes the classification head (`include_top=False`), freezes the convolutional base, adds Global Average Pooling → Dense(ReLU) → Dense(Softmax) |
| Task 3 – Model Training | Compiles and trains the frozen-base model (Adam, lr=0.001, batch size 32, 15 epochs, categorical cross-entropy) |
| Task 4 – Fine-Tuning | Unfreezes the last convolutional block of VGG16 (`block5_*`), retrains for 8 epochs at a lower learning rate (1e-5), compares accuracy before vs. after |
| Mandatory Plots (Accuracy & Loss) | Combined training/validation accuracy and loss curves spanning both the frozen and fine-tuning phases |
| Task 5 – Model Evaluation | Computes accuracy, precision, recall, F1-score, confusion matrix, classification report, and a sample of misclassified images on the final fine-tuned model |
| Hyperparameter Study | Trains 7 configurations (learning rate, batch size, optimizer, dense units, partial unfreezing) for 5 epochs each and ranks them by validation accuracy |
| Results | Final performance-metrics table and a comparison table of CNN architectures (LeNet-5, AlexNet, VGG16, GoogleNet, ResNet50) |
| Additional Exercise 2 | Repeats the transfer-learning pipeline with a ResNet50 backbone to populate the architecture-comparison table |
| Discussion | Answers the 10 conceptual questions from the lab manual (AlexNet, VGG16 filter choice, Inception module, residual learning, LeNet vs. ResNet, transfer learning, fine-tuning, dilated vs. transpose convolution, why pretrained models converge faster, computational complexity) |

4. Results
4.1 Final Model Performance (VGG16, Frozen + Fine-Tuned)

| Metric | Value |
|---|---|
| Training Accuracy (final epoch) | 98.86% |
| Testing Accuracy | 77.73% |
| Precision (macro) | 0.7805 |
| Recall (macro) | 0.7773 |
| F1-score (macro) | 0.7757 |
| Total Parameters | 14,781,642 |
| Trainable Parameters (frozen phase) | 66,954 |
| Total Training Time | 1086.6 sec (~18.1 min) — 580.5s frozen-base + 506.1s fine-tuning |

4.2 Before vs. After Fine-Tuning

| Stage | Test Accuracy |
|---|---|
| Frozen base only (after 15 epochs) | 65.90% |
| After fine-tuning last conv block (8 more epochs) | 77.73% |
| Improvement | +11.83 percentage points |

Observation: Fine-tuning the last convolutional block of VGG16 gave a substantially larger accuracy gain than continuing to train the frozen-base model, since it let the deepest, most task-specific filters adapt to CIFAR-10's statistics while the low/mid-level ImageNet features (edges, textures) were retained.

4.3 Hyperparameter Study (5-epoch quick tests, frozen base unless noted)

| Configuration | Validation Accuracy | Time (s) |
|---|---|---|
| Partial frozen base (last block unfrozen) | 0.7967 | 313.8 |
| Dense units = 256 | 0.6544 | 198.6 |
| Batch size = 16, Adam | 0.6537 | 227.1 |
| Batch size = 64, Adam | 0.6533 | 191.9 |
| LR = 0.001, Adam (baseline) | 0.6501 | 196.1 |
| LR = 0.0001, Adam | 0.6172 | 197.1 |
| SGD optimizer | 0.5919 | 196.2 |

Observation: Partially unfreezing the base clearly outperformed every frozen-base configuration even within a short 5-epoch budget, confirming that allowing some pretrained weights to adapt is more valuable than tuning learning rate, batch size, or dense-layer width alone. Adam consistently outperformed SGD given the same short training budget.

4.4 Per-Class Performance Highlights (from Confusion Matrix, out of 1000 per class)

| Class | Correctly Classified | Notes |
|---|---|---|
| Ship | 934 | Best performing class |
| Automobile | 848 | Very well classified |
| Truck | 836 | Strong performance |
| Airplane | 819 | Strong performance |
| Frog | 814 | Good performance |
| Horse | 805 | Good performance |
| Dog | 772 | Moderate performance, inflated by cat misclassifications |
| Bird | 738 | Moderate performance |
| Deer | 711 | Moderate performance |
| Cat | 496 | Weakest class; heavily confused with dog |

Most common confusions: cat → dog (249 cats misclassified as dog, the single largest off-diagonal cell), dog → cat (87), airplane → ship (82), horse → dog (79), deer → bird (64), truck → automobile (55). As with the earlier CNN-from-scratch experiment, rigid, geometrically distinct vehicle classes (ship, automobile, truck, airplane) were classified far more reliably than texture-heavy, pose-variable animal classes — though transfer learning substantially narrowed this gap compared to training from scratch.

4.5 Training Behavior Summary

| Curve | Behavior |
|---|---|
| Training Accuracy | Rises from ~0.56 to ~0.75 during the frozen phase, then jumps sharply to ~0.99 during fine-tuning |
| Validation Accuracy | Rises steadily to ~0.66 during the frozen phase, then climbs further to ~0.78 once fine-tuning begins, plateauing near epoch 6–7 of that phase |
| Training Loss | Decreases steadily through both phases, dropping sharply once fine-tuning starts (to ~0.06 by the final epoch) |
| Validation Loss | Decreases through the frozen phase, dips further early in fine-tuning (~epoch 2–3), then creeps back up slightly by the final epoch |

Key finding: Unlike training a CNN from scratch, the frozen-base phase showed training and validation accuracy tracking closely (little overfitting), since only ~67K parameters were being learned. Once the last convolutional block was unfrozen, training accuracy raced ahead of validation accuracy (98.86% vs. 77.73%), and validation loss began rising slightly in the final epochs — an early sign of mild overfitting during fine-tuning that would benefit from early stopping or regularization if training continued.

4.6 Architecture Comparison

| Model | Parameters | Accuracy (%) | Training Time (s) |
|---|---|---|---|
| LeNet-5 | 60K | – (reference/literature) | – |
| AlexNet | 61M | – (reference/literature) | – |
| VGG16 (this run) | 14,781,642 | 77.73 | 1086.6 |
| GoogleNet | 6.8M | – (reference/literature) | – |
| ResNet50 (this run) | 23,851,274 | 41.28 | 390.3 |

Observation: In this run, VGG16 substantially outperformed ResNet50 on CIFAR-10 within the same short (15-epoch, frozen-then-fine-tuned only for VGG16) training budget; ResNet50's batch-normalization statistics and deeper residual structure typically need more epochs (or a higher learning rate / partial fine-tuning) to adapt well to a small, low-resolution dataset like CIFAR-10, whereas VGG16's simpler feature hierarchy transferred more readily out of the box.

5. Mandatory Plots Generated
1. Sample CIFAR-10 Images (`plot1_sample_images.png`)
2. & 3. Training vs. Validation Accuracy (`plot2_3_accuracy.png`)
4. & 5. Training vs. Validation Loss (`plot4_5_loss.png`)
6. Confusion Matrix (`plot6_confusion_matrix.png`)
7. Misclassified Images (optional) (`plot7_misclassified.png`)
8. Hyperparameter Study Comparison (`plot8_hyperparameter_study.png`)

Each plot is accompanied by a short (2–3 line) inference in the notebook, in line with the lab manual's reporting requirement.

6. Conclusion
A transfer-learning pipeline built on a VGG16 backbone pretrained on ImageNet was successfully implemented on CIFAR-10: the convolutional base was frozen and used as a fixed feature extractor, a new classification head (Global Average Pooling → Dense → Softmax) was trained on top, and the last convolutional block was subsequently unfrozen and fine-tuned at a low learning rate. This progression raised test accuracy from 65.90% (frozen base only) to 77.73% (after fine-tuning) — an 11.83-point improvement — with a macro F1-score of 0.7757, using 14,781,642 total parameters and roughly 18 minutes of combined training time.

The experiment met all its stated learning outcomes: a pretrained CNN was adapted end-to-end (loading weights, removing the top, freezing/unfreezing layers), the effect of fine-tuning was quantified, a hyperparameter study ranked the relative importance of learning rate, batch size, optimizer choice, dense-layer width, and partial unfreezing, and the model was evaluated with a complete set of standard classification metrics. An additional ResNet50 run confirmed that transfer-learning performance is backbone-dependent within a fixed epoch budget.

Compared to training a CNN from scratch on CIFAR-10 (Experiment 3, 66.14% test accuracy with a shallow 2-conv-block network), transfer learning with a pretrained VGG16 backbone improved test accuracy by roughly 11–12 points using a comparable-or-smaller training time, while also narrowing — though not eliminating — the classic cat/dog and vehicle-vs-animal confusion patterns. The confusion matrix again shows the network handles rigid, shape-distinct object classes (ship, automobile, truck, airplane) more reliably than texture-heavy, pose-variable animal classes (cat, dog, deer, bird), with cat remaining the single weakest class.

Recommended improvements for future iterations:
- Early stopping during the fine-tuning phase, where validation loss first bottoms out, to curb the mild overfitting seen in later fine-tuning epochs.
- Progressive unfreezing of additional convolutional blocks (beyond just `block5`) at even lower learning rates.
- Data augmentation (random crops, flips, rotations) to further improve generalization on the confused animal classes.
- A more exhaustive hyperparameter sweep (longer epoch budgets per configuration) to validate the ranking found in the quick 5-epoch study.
- Repeating the pipeline with additional pretrained backbones (e.g., EfficientNet, MobileNet) to extend the architecture-comparison table with more empirically measured (rather than literature-reference) rows.

Overall, the experiment illustrates the core strength of transfer learning — reusing generic, large-scale pretrained features to reach strong accuracy on a small dataset with far less training than learning from scratch — as well as the practical trade-offs involved in choosing how much of a pretrained network to fine-tune.

7. References
- Goodfellow, I., Bengio, Y., & Courville, A. — Deep Learning
- Bishop, C. M. — Pattern Recognition and Machine Learning
- Haykin, S. — Neural Networks and Learning Machines
- Simonyan, K., & Zisserman, A. — Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG16)
- He, K., Zhang, X., Ren, S., & Sun, J. — Deep Residual Learning for Image Recognition (ResNet)
- TensorFlow Documentation — https://www.tensorflow.org/
- Keras Applications (Pretrained Models) — https://keras.io/api/applications/
- CIFAR-10 Dataset Documentation — https://www.cs.toronto.edu/~kriz/cifar.html
