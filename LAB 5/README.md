# CS3807 – Deep Learning Laboratory

## Experiment 5: Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

**Institution:** Shiv Nadar University Chennai **Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V **Subject Code:** CS3807 – Deep Learning Laboratory **Academic Year:** 2026–27 **Model:** MobileNetV2 **Dataset:** Oxford-IIIT Pet Dataset

## 1. Objective

To conduct an end-to-end empirical study of the factors that govern CNN training performance, using MobileNetV2 transfer learning on a real-world image classification task, by implementing:

- Weight initialization comparison (Zero, Random, Xavier/Glorot, He)
- Regularization and overfitting analysis (No regularization, L2, Dropout, Batch Normalization)
- A from-scratch numerical verification of the Batch Normalization formula
- Optimizer comparison (SGD, Momentum, RMSProp, Adam)
- Hyperparameter tuning (learning rate, batch size, dropout rate) changing one variable at a time
- Transfer learning via feature extraction vs. fine-tuning of a pretrained MobileNetV2 backbone
- Model selection using 5-fold cross-validation (mean **and** standard deviation)
- Final evaluation on a held-out, untouched test set with full classification metrics
- An additional exercise proposing and cross-validating two new candidate configurations

> **Note on execution:** This notebook is written to run end-to-end on Google Colab or a machine with internet access and, ideally, a GPU (**Runtime → Change runtime type → GPU** on Colab). Every experiment cell is followed by its required plot(s) and a short **Inference** cell with `[...]` placeholders — these are filled in automatically with your run's actual numbers once the notebook is executed top to bottom. Epoch counts are deliberately kept small (6–15) to keep total runtime manageable; increase them for smoother curves if GPU time allows.

## 2. Dataset Description

Dataset: Oxford-IIIT Pet Dataset (via `tensorflow_datasets`)

| Property | Value |
|---|---|
| Number of classes | 37 (pet breeds — cats and dogs) |
| Provided splits | `train`, `test` |
| Train/validation split | `train` split further divided 80% / 20% |
| Test split | Kept completely untouched until Section 12 |
| Image size (native) | Variable, resized to 224 × 224 × 3 |
| Preprocessing | `tf.keras.applications.mobilenet_v2.preprocess_input` (scales pixels to [-1, 1]) |
| Augmentation (where used) | Random horizontal flip, random brightness |

## 3. Contents of the Notebook (CS3807_Experiment5_MobileNetV2_PetClassification.ipynb)

The notebook is organized into the following sections, matching the lab manual's task structure:

| Section | Description |
|---|---|
| 0. Setup and Imports | Imports TensorFlow/Keras, NumPy, pandas, Matplotlib, `tensorflow_datasets`, scikit-learn; sets random seeds and plot style for reproducibility |
| 3. Dataset and Experimental Setup | Loads Oxford-IIIT Pet via `tfds`, builds the train/validation/test `tf.data` pipelines with MobileNetV2 preprocessing |
| 4. MobileNetV2 Architecture — Model Builder | A single reusable `build_model()` function (init, dropout, batchnorm, L2, frozen/fine-tuned base, optimizer, learning rate) so every later section only changes the parameter(s) under study |
| 5. Weight Initialization | Compares Zero, Random, Xavier/Glorot, and He initialization on the classifier head (Plots 1–2) |
| 6. Regularization and Overfitting | Compares No Regularization, L2, Dropout, and Batch Normalization (Plots 3–4) |
| 7. Batch Normalization | Numerical worked example of the BatchNorm formula, verified against Keras's own layer; With-vs-without BN comparison (Plot 5) |
| 8. Optimization Algorithms | Compares SGD, Momentum, RMSProp, and Adam with an optimizer comparison table (Plots 6–7) |
| 9. CNN Hyperparameter Tuning | One-hyperparameter-at-a-time sweep over learning rate, batch size, and dropout rate (Plots 8–10) |
| 10. Transfer Learning and Fine-Tuning | Case A (frozen-base feature extraction) vs. Case B (fine-tuning the upper base layers at a low learning rate) (Plots 11–12) |
| 11. K-Fold Cross-Validation | 5-fold CV over four candidate configurations (C1–C4) informed by Sections 5–10, reported as mean ± SD (Plot 13) |
| 12. Final Model Evaluation | Best CV configuration retrained on the full train+val pool and evaluated once on the untouched test set: accuracy, precision, recall, F1-score, confusion matrix (Plot 14), and misclassified images (Plot 15) |
| 13. Overall Results | Summary table tracking the best result achieved at each stage of the study (initialization → regularization → optimizer → hyperparameters → transfer learning → final model) |
| 14. Required Inference for Plots | Confirms that a 2–3 line inference (what it shows / trend observed / why) accompanies every plot (1–15) |
| 15. Discussion Questions | Answers 23 conceptual questions covering parameters vs. hyperparameters, initialization, regularization, Batch Normalization, optimizers, learning rate/batch size effects, stride/padding, MobileNetV2 efficiency, transfer learning, and cross-validation |
| 16. Additional Exercise | Proposes two new configurations (aggressive fine-tuning with low dropout; frozen-base with high dropout and RMSProp), cross-validates them, and compares against the previously selected best configuration |
| 17. Expected Outcome | Summarizes how each stage of the study (initialization, regularization, optimization, hyperparameters, transfer learning, cross-validation) is expected to shape final model performance |

## 4. Methodology Highlights

**4.1 Reusable Model Builder (Section 4)**
A single `build_model()` function parameterizes initializer, dropout rate, batch normalization, L2 regularization, frozen/fine-tuned base (with an optional `fine_tune_at` layer index), optimizer, and learning rate — so each experiment section isolates exactly one variable, per the lab manual's "change one hyperparameter at a time" rule.

**4.2 Weight Initialization (Section 5)**
Zero, Random (`RandomNormal`), Xavier/Glorot, and He initialization are compared on the classifier head with the pretrained base frozen, isolating the effect of initialization on convergence speed and final validation accuracy.

**4.3 Regularization (Section 6) and Batch Normalization (Section 7)**
No Regularization, L2, Dropout, and Batch Normalization are compared for their effect on the training/validation gap. Section 7 additionally reproduces the manual's numerical Batch Normalization example (for x = [2, 4, 6, 8]) by hand and cross-checks it against Keras's `BatchNormalization` layer.

**4.4 Optimizers (Section 8)**
SGD, Momentum, RMSProp, and Adam are compared under identical architecture and learning rate, with a results table capturing final loss, best validation accuracy, epoch-to-converge, and wall-clock training time for each.

**4.5 Hyperparameter Tuning (Section 9)**
Learning rate ({0.001, 0.0001}), batch size ({16, 32, 64}), and dropout rate ({0.0, 0.25, 0.5}) are each swept independently against a fixed baseline configuration.

**4.6 Transfer Learning and Fine-Tuning (Section 10)**
Case A freezes the entire MobileNetV2 base and trains only the new head (feature extraction). Case B unfreezes the base from a chosen layer index (`fine_tune_at = 100`) onward and retrains jointly at a much smaller learning rate (1e-5), directly comparing the two strategies.

**4.7 Cross-Validation and Final Selection (Sections 11–12)**
Four candidate configurations (C1–C4), informed by the best settings found in Sections 5–10, are evaluated with 5-fold cross-validation on the training pool only (test set untouched). The configuration with the highest mean CV accuracy is retrained on the full train+validation pool and evaluated exactly once on the independent test set, reporting accuracy, precision, recall, F1-score, a full 37-class confusion matrix, and the top confused breed pairs.

**4.8 Additional Exercise (Section 16)**
Two new configurations — an aggressive fine-tune with low dropout (`fine_tune_at = 50`) and a high-dropout frozen-base configuration with RMSProp — are cross-validated the same way and compared against the previously selected best configuration on accuracy, stability (SD), and computational cost.

## 5. Mandatory Plots Generated

1. Training Loss vs. Epoch — Weight Initialization (`plot01_init_train_loss`)
2. Validation Accuracy vs. Epoch — Weight Initialization (`plot02_init_val_acc`)
3. Training & Validation Accuracy — Regularization (`plot03_reg_accuracy`)
4. Training & Validation Loss — Regularization (`plot04_reg_loss`)
5. With vs. Without Batch Normalization (`plot05_bn_comparison`)
6. Training Loss vs. Epoch — Optimizers (`plot06_optimizer_train_loss`)
7. Validation Accuracy vs. Epoch — Optimizers (`plot07_optimizer_val_acc`)
8. Learning Rate vs. Validation Accuracy (`plot08_lr_vs_acc`)
9. Batch Size vs. Validation Accuracy (`plot09_batchsize_vs_acc`)
10. Dropout Rate vs. Validation Accuracy (`plot10_dropout_vs_acc`)
11. Feature Extraction vs. Fine-Tuning (`plot11_fe_vs_ft`)
12. Training/Validation Loss — Before vs. After Fine-Tuning (`plot12_finetune_loss`)
13. 5-Fold Cross-Validation Accuracy ± SD (`plot13_cv_accuracy`)
14. Confusion Matrix — 37 Pet Breeds (`plot14_confusion_matrix`)
15. Misclassified Images (optional) (`plot15_misclassified`)
16. Additional Exercise Comparison (`plot_additional_exercise_comparison`)

Each plot is accompanied by a short (2–3 line) inference directly beneath it, covering *what the plot shows*, *the trend observed*, and *why it likely occurs*, in line with the lab manual's reporting requirement. Figures are saved as both PNG (viewing) and 600 dpi EPS (report submission) into a `figures/` directory.

## 6. Results

This notebook is provided as a **template with the full experimental pipeline wired up but not yet executed** — every Inference cell contains bracketed `[...]` placeholders (e.g. expected relative trends like "Zero initialization is expected to stagnate at [...]% validation accuracy") that are meant to be replaced with the actual numbers your run produces. Running all cells top to bottom (ideally on a GPU runtime) will:

- Populate the Weight Initialization, Regularization, Optimizer, and Hyperparameter comparison plots and tables with real numbers (Sections 5–9)
- Report the measured accuracy gain from fine-tuning vs. feature extraction (Section 10)
- Produce the 5-fold CV mean ± SD table for configurations C1–C4 and select a best configuration (Section 11)
- Report final test accuracy, precision, recall, F1-score, the full 37×37 confusion matrix, and the most-confused breed pairs for the selected configuration (Section 12)
- Fill in the Overall Results summary table tracking the best result at each stage (Section 13)
- Report the additional exercise's comparison of two new configurations (E1, E2) against the originally selected configuration (Section 16)

Once executed, this section should be updated with the actual measured values (e.g. best initializer and its validation accuracy, best regularization strategy, best optimizer, optimal learning rate/batch size/dropout, feature-extraction vs. fine-tuned accuracy, the winning cross-validated configuration with its mean ± SD, and final test-set metrics) so the README reflects the specific run's results rather than the template's expected trends.

## 7. Discussion Questions

The notebook answers 23 conceptual questions from the lab manual, grouped by topic:

- **Parameters vs. hyperparameters, and initialization** (Q1–Q4): the learned-vs-set distinction, why initialization matters, why zero initialization fails via symmetry, and a comparison of Xavier vs. He initialization for ReLU networks.
- **Overfitting, Dropout, and Batch Normalization** (Q5–Q9): diagnosing overfitting from training/validation curves, how Dropout acts as an implicit ensemble, the purpose of Batch Normalization, a worked numerical example, and the role of the learnable γ/β parameters.
- **Optimizers and learning dynamics** (Q10–Q13): a comparison of SGD, Momentum, RMSProp, and Adam, and the effects of learning rate and batch size that are too large, too small, or too big.
- **Convolutional mechanics and MobileNetV2 efficiency** (Q14–Q16): stride and padding, and why depthwise separable convolutions and inverted residual bottlenecks make MobileNetV2 lightweight.
- **Transfer learning** (Q17–Q19): the definition of transfer learning, feature extraction vs. fine-tuning, and why fine-tuning uses a smaller learning rate.
- **Cross-validation and model selection** (Q20–Q23): why K-fold CV gives a more reliable estimate than a single split, why the test set must remain untouched during tuning, why both mean and standard deviation matter, and why the highest validation accuracy alone isn't sufficient to select a model.

## 8. Expected Outcome

The study is designed to demonstrate, section by section, that MobileNetV2's performance on Oxford-IIIT Pet classification is materially shaped by weight initialization (He/Xavier vs. zero/naive random), regularization choices (Dropout, L2, Batch Normalization vs. none), optimizer choice (SGD/Momentum/RMSProp/Adam), and hyperparameter settings (learning rate, batch size, dropout rate). Building on these findings, fine-tuning the upper pretrained layers at a small learning rate is expected to improve on plain feature extraction by letting MobileNetV2's ImageNet-general features specialize to pet breeds. Finally, a single reliable final configuration is selected and justified using 5-fold cross-validation (mean **and** standard deviation, not accuracy alone), confirmed once on an untouched independent test set, with per-class performance and failure modes examined via the confusion matrix and misclassified-image inspection. The additional exercise further tests whether this selection process generalizes to new candidate configurations beyond the original four.

## 9. References

- Goodfellow, I., Bengio, Y., & Courville, A. — Deep Learning
- Bishop, C. M. — Pattern Recognition and Machine Learning
- Haykin, S. — Neural Networks and Learning Machines
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. — MobileNetV2: Inverted Residuals and Linear Bottlenecks
- Ioffe, S., & Szegedy, C. — Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- Kingma, D. P., & Ba, J. — Adam: A Method for Stochastic Optimization
- TensorFlow Documentation — https://www.tensorflow.org/
- TensorFlow Datasets: Oxford-IIIT Pet — https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet
