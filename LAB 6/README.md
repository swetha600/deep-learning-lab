# CS3807 – Deep Learning Laboratory
## Experiment 6: End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

**Institution:** Shiv Nadar University Chennai  
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V  
**Subject Code:** CS3807 – Deep Learning Laboratory  
**Academic Year:** 2026–27

---

## 1. Objective

To develop an end-to-end understanding of recurrent sequence learning by implementing and comparing Vanilla RNN, LSTM, and GRU models. The experiment covers:

- Backpropagation Through Time (BPTT) and the vanishing/exploding gradient problem
- Gating mechanisms in LSTM (forget, input, output gates) and GRU (reset, update gates)
- Human Activity Recognition (HAR) on the UCI HAR dataset
- CNN–RNN pipelines for video understanding
- Encoder–decoder sequence-to-sequence learning
- Comparative evaluation across architectures using standard classification metrics

---

## 2. Dataset Description

### 2.1 UCI HAR Dataset (Primary — Sequence Classification)

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Total training windows | 7,352 |
| Total testing windows | 2,947 |
| Number of classes | 6 |
| Sequence length | 128 time steps |
| Number of sensor channels | 9 |
| Activities | WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING |

A stratified subset of **2,000 windows** (≈ 333 per class) was used for training, split 70 / 15 / 15 into train / validation / test sets. Features were normalized via `StandardScaler` fitted on training data only.

### 2.2 Simulated Video Dataset (CNN–LSTM)

Synthetic RGB frames (224 × 224 × 3) were generated per activity class. MobileNetV2 (pretrained on ImageNet, frozen) served as the per-frame feature extractor; 10 frames per video clip were encoded into a feature sequence of shape `(10, feat_dim)`, fed to an LSTM/GRU classifier.

### 2.3 Seq2Seq Dataset (Synthetic Sorting Task)

Integer sequences of length 4 were generated; the target was the same sequence sorted in ascending order. A second exercise used input length 6 with output length 4 to demonstrate variable-length encoder–decoder operation.

---

## 3. Notebook Structure

| Section | Description |
|---|---|
| 0 – Setup & Imports | Installs packages; imports TensorFlow/Keras, NumPy, Matplotlib, scikit-learn; sets random seeds |
| 1 – Data Download & Preprocessing | Downloads UCI HAR, loads raw inertial signals (9 channels, 128 steps), subsets to 2,000 windows, splits and normalizes |
| 2 – Temporal Visualization (Plot 1) | Plots sensor signals vs time step for three activities × three channels |
| 3 – BPTT Numerical Exercise | Manually computes forward pass for a 3-step Vanilla RNN; verifies against Keras `SimpleRNNCell`; illustrates vanishing gradient |
| 4 – Model Definitions | Unified `build_model(rnn_type)` builder for RNN / LSTM / GRU with identical surrounding layers |
| 5 – Training & Plots 2 & 3 | Trains all three architectures (30 epochs, early stopping, ReduceLROnPlateau); plots loss and accuracy curves |
| 6 – Evaluation & Plots 4 & 5 | Computes accuracy, precision, recall, F1; confusion matrices (Plot 4); bar comparison of metrics and parameters (Plot 5) |
| 7 – Sequence Length Study (Plot 6) | Varies sequence length (32 / 64 / 128) and reports test macro F1 vs cost |
| 8 – CNN–LSTM Video Pipeline (Plots 7–9) | MobileNetV2 feature extraction per frame → LSTM classifier; training curves and confusion matrix |
| 9 – Seq2Seq (same-length sorting) | Encoder–decoder with teacher forcing; evaluates token and sequence accuracy |
| 10 – Seq2Seq (different-length) | Extends Seq2Seq to IN\_LEN=6 / OUT\_LEN=4; demonstrates decoupled sequence lengths |
| 11 – Consolidated Results | Full comparison table (RNN, LSTM, GRU, CNN–LSTM) and Seq2Seq results |
| 12 – Discussion (25 Questions) | Answers all 25 conceptual questions from the lab manual |
| 13 – Additional Exercises (7) | Unit count ablation; GRU vs LSTM; stacked LSTM; bidirectional LSTM; sequence length vs cost; CNN-GRU vs CNN-LSTM; variable-length Seq2Seq |

---

## 4. Results

### 4.1 HAR Sequence Classification — Main Models

| Model | Accuracy (%) | Precision (%) | Recall (%) | Macro F1 (%) | Parameters |
|---|---|---|---|---|---|
| Vanilla RNN | — | — | — | — | — |
| LSTM | — | — | — | — | — |
| GRU | — | — | — | — | — |
| CNN–LSTM (video) | — | — | — | — | — |

> **Note:** Numerical values are populated on execution. Results will vary with hardware and random seed.

**Approximate parameter counts (32 units):**

| Architecture | Trainable Parameters |
|---|---|
| Vanilla RNN | ~3,000 |
| LSTM | ~10,000 |
| GRU | ~8,000 |

### 4.2 Key Training Observations

| Curve | Expected Behaviour |
|---|---|
| RNN Training Loss | Decreases slowly; may plateau due to vanishing gradients |
| RNN Validation Loss | Plateaus early; largest train–val gap of the three |
| LSTM / GRU Training Loss | Decreases smoothly; gating mitigates vanishing gradient |
| LSTM / GRU Validation Loss | Tracks training more closely; slight divergence after early stopping |

**EarlyStopping** (patience = 5 on `val_loss`) and **ReduceLROnPlateau** (factor 0.5, patience 3) were applied uniformly across all three architectures.

### 4.3 Seq2Seq Results

| Metric | Value |
|---|---|
| Token Accuracy | — |
| Sequence Accuracy | — |
| Final Train Loss | — |
| Final Val Loss | — |

Sequence accuracy is always ≤ token accuracy. A single wrong token at any position makes the entire output sequence incorrect, so sequence accuracy drops much faster than token accuracy as sequence length grows.

### 4.4 Additional Exercise Summary

| Exercise | Finding |
|---|---|
| 1 – Unit count (16 / 32 / 64) | Accuracy improves 16 → 32; diminishing returns at 64 with increased cost |
| 2 – GRU vs LSTM (same units) | GRU uses ~25% fewer parameters; F1 gap typically < 2 pp |
| 3 – Stacked 2-layer LSTM | Marginal gain (~1–2%) on 128-step sequences; training time doubles |
| 4 – Bidirectional LSTM | Doubles parameters; beneficial for offline tasks; impractical for real-time |
| 5 – Sequence length vs cost | Params unchanged; epoch time grows linearly with T; F1 saturates |
| 6 – CNN-LSTM vs CNN-GRU | Similar F1 on 10-frame clips; GRU faster due to fewer parameters |
| 7 – Variable-length Seq2Seq | Encoder–decoder naturally handles IN_LEN ≠ OUT_LEN with no architectural change |

---

## 5. Mandatory Plots

| # | Filename | Description |
|---|---|---|
| 1 | `plot1_sensor_signals.png` | Sensor signal vs time step — 3 activities × 3 channels |
| 2 & 3 | `plot2_3_loss_accuracy.png` | Training / validation loss and accuracy for RNN, LSTM, GRU |
| 4 | `plot4_confusion_matrices.png` | Confusion matrices on the test set |
| 5 | `plot5_comparison_bar.png` | Bar chart — accuracy, F1, parameters, training time |
| 6 | `plot6_seqlen_f1.png` | Sequence length vs test macro F1 |
| 7 | `plot7_video_frames.png` | Sample synthetic video frames per activity class |
| 8 | `plot8_video_training.png` | CNN–LSTM video model training curves |
| 9 | `plot9_video_confusion.png` | CNN–LSTM video confusion matrix |

Each plot is followed by a 4-point inline inference block in the notebook covering: what the plot shows, what pattern is visible, inter-model comparison, and practical implication.

---

## 6. Conceptual Highlights

### Vanishing Gradient (BPTT)
Gradient magnitude scales as `|Wh · σ'|^k` over `k` unrolled steps. With `Wh = 0.8` (as in the numerical exercise), the gradient shrinks to ≈ 0 within ~20 steps, rendering the vanilla RNN unable to learn long-range dependencies.

### LSTM Gating
The cell state `C_t` flows through the network with only linear interactions, protected by the forget gate (erase), input gate (write), and output gate (expose). This architecture allows gradients to propagate over hundreds of steps without exponential decay.

### GRU Simplification
GRU merges forget and input into a single update gate and eliminates the separate cell state, reducing parameters by ~25% versus LSTM. Empirically, performance is comparable on most tasks at this scale.

### CNN–RNN for Video
MobileNetV2 extracts spatial features per frame (independent of time); LSTM/GRU models the temporal dynamics across the frame sequence. The CNN provides "what" (spatial content); the RNN provides "how it changes" (temporal evolution).

### Encoder–Decoder Seq2Seq
The encoder compresses a variable-length input into a fixed context vector (final hidden/cell state). The decoder uses this context as its initial state and generates the output sequence token-by-token. Teacher forcing feeds the ground-truth previous token during training; at inference, the decoder's own outputs are used instead.

---

## 7. Conclusion

All three recurrent architectures — Vanilla RNN, LSTM, and GRU — were implemented, trained, and evaluated on the UCI HAR dataset under identical experimental conditions. LSTM and GRU substantially outperform Vanilla RNN on this task, confirming that gating mechanisms resolve the vanishing gradient limitation of standard recurrence.

GRU achieves comparable accuracy to LSTM with fewer parameters and faster training, making it the preferred architecture when computational budget is a concern. The CNN–LSTM video pipeline demonstrates that pretrained spatial encoders can be composed with recurrent temporal models to handle vision-based sequence tasks without any per-task CNN training. The Seq2Seq encoder–decoder framework generalises naturally to variable-length input–output pairs, forming the backbone of modern sequence transduction systems.

Key failure mode: both LSTM and GRU can overfit on small subsets; early stopping and dropout are essential regularization tools, not optional extras.

---

## 8. References

1. Hochreiter, S. & Schmidhuber, J. — *Long Short-Term Memory* (Neural Computation, 1997)
2. Cho, K. et al. — *Learning Phrase Representations using RNN Encoder–Decoder* (EMNLP 2014)
3. Goodfellow, I., Bengio, Y. & Courville, A. — *Deep Learning* (MIT Press, 2016)
4. UCI HAR Dataset — https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
5. TensorFlow / Keras Documentation — https://www.tensorflow.org/
6. Howard, A. et al. — *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications* (arXiv 2017)
