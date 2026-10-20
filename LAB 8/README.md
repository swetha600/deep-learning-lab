# CS3807 – Deep Learning Laboratory

## Experiments 8 & 9: Effect of Attention Type and Attention Position in a Pretrained ResNet50

**Institution:** Shiv Nadar University Chennai
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V
**Subject Code:** CS3807 – Deep Learning Laboratory
**Academic Year:** 2026–27

## 1. Objective

To understand how different attention mechanisms modify a pretrained convolutional network, and how the position at which attention is inserted influences image classification performance. A single backbone, ImageNet-pretrained ResNet50, is used throughout. The experiment covers:

- Channel attention (SE-style), spatial attention, and sequential channel–spatial attention
- Lightweight self-attention over the spatial positions of a feature map
- Insertion of attention at different ResNet50 stages (P1–P4), at single and multiple stages
- Attention ordering (Channel→Spatial vs. Spatial→Channel) and spatial-kernel size (3/5/7)
- Additional established mechanisms: SE, ECA and CBAM
- Quantitative evaluation using accuracy, macro precision/recall/F1, parameters and inference time
- Qualitative interpretation using internal attention maps and Grad-CAM

## 2. Dataset Description

### 2.1 TensorFlow Flowers (`tf_flowers`)

| Property | Value |
|---|---|
| Source | TensorFlow Datasets `tf_flowers` |
| Image dimensions | 224 × 224 × 3 (RGB, resized) |
| Classes | 5 (daisy, dandelion, roses, sunflowers, tulips) |
| Split (train : val : test) | 70% : 15% : 15% → 2,569 / 550 / 551 images |
| Test class counts | 126 dandelion, 98 daisy, 118 tulips, 111 sunflowers, 98 roses |
| Preprocessing | ResNet50 ImageNet `preprocess_input` |
| Augmentation | Random flip, small rotation, zoom, translation, mild contrast |

The same split, preprocessing and augmentation are reused for every model so that differences in performance come only from the factor under study.

### 2.2 Candidate Attention Positions

| Position | Location | Resolution | Channels |
|---|---|---|---|
| P1 | After conv2_x | 56 × 56 | 256 |
| P2 | After conv3_x | 28 × 28 | 512 |
| P3 | After conv4_x | 14 × 14 | 1024 |
| P4 | After conv5_x | 7 × 7 | 2048 |

All attention modules preserve the tensor shape (input and output H × W × C are identical).

### 2.3 Training Protocol

| Setting | Value |
|---|---|
| Backbone | ResNet50 (ImageNet), classifier = GAP → Dropout(0.3) → Dense(5) |
| Optimizer / loss | Adam / categorical cross-entropy |
| Stage 1 | LR 1e-3, 3 epochs, backbone frozen |
| Stage 2 | LR 1e-4, 2 epochs, conv5_x unfrozen |
| Batch size / seed | 16 / 42 |
| Callbacks | Early stopping on val accuracy (patience 5, best weights restored) |
| Inference time | Mean of 50 single-image forward passes |

Note: the notebook was run in its QUICK RUN mode (5 epochs in total) rather than the full 25-epoch setting, so all results come from a short schedule and a single run per model.

## 3. Notebook Structure

| Section | Description |
|---|---|
| 0 – Setup & Configuration | Imports, seeds, hyper-parameters, QUICK RUN switch |
| 1 – Dataset (Viz #1) | Loads `tf_flowers`, fixed 70/15/15 split, sample image per class |
| 2 – ResNet50 Backbone | Extracts stage outputs conv2_x–conv5_x for attention insertion |
| 3 – Attention Modules | Channel (SE-style), Spatial (7×7), Channel→Spatial and Spatial→Channel, lightweight Self-Attention, ECA, CBAM |
| 4 – Model Builder | Inserts any mechanism at any stage or combination of stages |
| 5 – Training & Evaluation Utilities | Two-stage training, metrics, timing, checkpoint/resume support |
| 6 – Baseline (A0) | Pretrained ResNet50 without attention |
| 7 – Experiment 1 | Attention-type ablation at P3 (A0–A5) |
| 8 – Experiment 2 | Channel–Spatial position ablation (B1–B4) |
| 9 – Experiment 3 | Multiple positions: P3, P4, P3+P4, P2+P3+P4 |
| 10 – Experiment 4 | Attention order at P3 |
| 11 – Experiment 5 | Spatial kernel size 3/5/7 |
| 12 – Additional Task | SE, ECA and CBAM comparison |
| 13 – Required Main Result Table | Consolidated results table |
| 14 – Graphical Analysis (Plots 2–7) | Accuracy/loss curves, accuracy and F1 bar charts, accuracy vs. parameters, F1 vs. inference time |
| 15 – Confusion Matrices | Baseline vs. Channel+Spatial (P3) |
| 16 – Grad-CAM | Baseline and attention-model Grad-CAM |
| 17 – Internal Attention Maps | Channel weights (top-20), spatial heatmap/overlay, self-attention response, feature maps before/after attention |
| 18 – Attention Position Visualization | Spatial attention at P2, P3, P4 for one image |
| 19 – Qualitative Comparison | Five test images (three correct, two incorrect for the baseline) |
| 20 – Discussion | Answers to the 35 manual questions |
| 21 – Save All Results | Exports metrics and figures |

## 4. Results

### 4.1 Attention-Type Ablation (all at P3)

| Model | Accuracy | Precision | Recall | Macro F1 | Total Params | Time (ms) |
|---|---|---|---|---|---|---|
| A0 Baseline | 0.9328 | 0.9358 | 0.9305 | 0.9323 | 23,597,957 | 95.46 |
| A1 Channel | 0.9365 | 0.9363 | 0.9377 | 0.9359 | 23,730,117 | 90.41 |
| A2 Spatial (7×7) | 0.9256 | 0.9304 | 0.9223 | 0.9248 | 23,598,056 | 109.36 |
| A3 Channel–Spatial | 0.9328 | 0.9341 | 0.9312 | 0.9316 | 23,730,216 | 96.77 |
| A4 Self-Attention | **0.9437** | 0.9428 | 0.9446 | **0.9435** | 24,909,958 | 115.30 |
| A5 Combined | 0.9347 | 0.9375 | 0.9319 | 0.9340 | 25,042,217 | 97.22 |

### 4.2 Attention-Position Ablation (Channel→Spatial)

| Position | Accuracy | Macro F1 | Total Params | Time (ms) |
|---|---|---|---|---|
| P1 | 0.9093 | 0.9085 | 23,606,520 | 106.76 |
| P2 | 0.9147 | 0.9140 | 23,631,368 | 104.28 |
| **P3** | **0.9328** | 0.9316 | 23,730,216 | 96.77 |
| P4 | 0.9183 | 0.9166 | 24,124,520 | 105.93 |

### 4.3 Multiple Attention Positions

| Configuration | Accuracy | Macro F1 | Total Params |
|---|---|---|---|
| P3 only | 0.9328 | 0.9316 | 23,730,216 |
| P4 only | 0.9183 | 0.9166 | 24,124,520 |
| P3 + P4 | 0.9020 | 0.9013 | 24,256,779 |
| P2 + P3 + P4 | 0.9256 | 0.9249 | 24,290,190 |

### 4.4 Attention Order and Spatial Kernel Size

| Experiment | Setting | Accuracy | Macro F1 | Time (ms) |
|---|---|---|---|---|
| Order (P3) | Channel → Spatial | 0.9328 | 0.9316 | 96.77 |
| Order (P3) | Spatial → Channel | 0.9310 | 0.9299 | 114.68 |
| Kernel | 3 × 3 | 0.9165 | 0.9157 | 91.20 |
| Kernel | 5 × 5 | 0.9129 | 0.9119 | 103.93 |
| Kernel | 7 × 7 | 0.9256 | 0.9248 | 109.36 |

### 4.5 Additional Mechanisms (P3)

| Mechanism | Accuracy | Macro F1 | Added Params | Time (ms) |
|---|---|---|---|---|
| No Attention | 0.9328 | 0.9323 | – | 95.46 |
| Channel / SE | 0.9365 | 0.9359 | 132,160 | 90.41 |
| Spatial | 0.9256 | 0.9248 | 99 | 109.36 |
| ECA | 0.9111 | 0.9106 | 5 | 103.11 |
| CBAM | 0.9074 | 0.9059 | 132,259 | 106.64 |
| Self-Attention | 0.9437 | 0.9435 | 1,312,001 | 115.30 |

### 4.6 Key Observations

| Topic | Observation |
|---|---|
| Best model | Self-attention at P3 (+1.09 points accuracy); channel attention is next (+0.36 points) |
| Significance | With 551 test images (one image = 0.18 points) the error is about ±2 points; McNemar tests give p = 0.36 (self-attention vs. baseline) and p = 1.0 (Channel+Spatial vs. baseline), so most gaps are not meaningful |
| Position | P3 is the best position for Channel–Spatial; P1 is the worst |
| Multiple stages | More attention locations did not help; P3+P4 was 3.1 points below P3 alone |
| Order / kernel | Differences are at most 1.3 points, within single-run noise |
| Complexity | More parameters do not guarantee higher accuracy; the 132 k channel module matches or beats the larger Combined model |
| Convergence | Self-attention started faster (first-epoch train acc. 0.771 vs. 0.745); no attention model ended with a lower validation loss than the baseline (0.221) |
| Timing | Only self-attention is clearly slower (~21%); other differences are within timing noise |

### 4.7 Confusion Matrix Summary (Baseline vs. Channel+Spatial P3)

| Item | Finding |
|---|---|
| Overall | Both classify 514 of 551 images correctly (0.9328) |
| Best class | Tulips (recall 0.966 → 0.983) |
| Most common confusion | Roses predicted as tulips (7 → 9 images) |
| Gains | Tulips (F1 0.912 → 0.943), sunflowers (0.932 → 0.942), daisy recall (0.908 → 0.939) |
| Losses | Dandelion (F1 0.964 → 0.939), roses recall (0.898 → 0.867) |

### 4.8 Qualitative Summary

| Visualization | Finding |
|---|---|
| Baseline Grad-CAM | On the seed head for the dandelion; on sunflowers and nearby sky for the sunflower image; spread out for the two cluttered-field errors |
| Channel+Spatial Grad-CAM | Almost uniform with a small corner spot; taken at the spatial-attention output layer, so **not comparable** with the baseline map |
| Channel weights | Top-20 weights are all ≈ 1.0 (saturated sigmoid) |
| Spatial maps (P2/P3/P4) | Nearly uniform; variation only at image borders, consistent with a zero-padding effect |
| Self-attention | Mean response concentrated on a single central location for the test image |
| Feature maps | Almost unchanged before vs. after attention, matching the unchanged baseline accuracy |

## 5. Mandatory Visualizations

| # | Description |
|---|---|
| 1 | Sample images from all five classes |
| 2 | Training and validation accuracy curves |
| 3 | Training and validation loss curves |
| 4 | Attention mechanism vs. accuracy / macro F1 |
| 5 | Accuracy vs. model parameters |
| 6 | Macro F1 vs. inference time |
| 7 | Confusion matrices (baseline and attention model) |
| 8 | Feature maps before and after attention |
| 9 | Channel-attention weights, spatial heatmap and overlay, self-attention response |
| 10 | Baseline and attention-model Grad-CAM; spatial attention maps at P2, P3, P4 |

Each plot is followed by an inline inference describing what is visible, the trend, and a cautious conclusion.

## 6. Conceptual Highlights

**ResNet50 Feature Hierarchy**
Spatial resolution falls (56 → 7) while channels rise (256 → 2048) with depth. Early stages encode edges, colours and textures; intermediate stages encode parts and shapes; late stages encode class-level semantics. The same attention module therefore behaves differently depending on where it is inserted.

**Channel Attention**
`M_c(F) = σ(W₂ δ(W₁ z))` with `z` the global-average-pooled descriptor and reduction ratio r = 16. It answers *which channels matter*.

**Spatial Attention**
`M_s(F) = σ(Conv₇ₓ₇([F_avg ‖ F_max]))`. It answers *which locations matter*.

**Sequential Channel–Spatial**
`F_cs = M_s(F_c) ⊙ F_c`, with `F_c = M_c(F) ⊙ F`: first *what*, then *where*.

**Self-Attention**
`Y = softmax(QKᵀ/√d)V` with residual `F + γ·Y`. The N × N attention matrix (N = HW) grows quadratically with the number of positions, so it is applied at P3 or P4.

**Internal Attention vs. Grad-CAM**
An attention module computes weights during the forward pass; Grad-CAM is a post-hoc, gradient-based explanation that does not alter the forward computation.

## 7. Conclusion

All attention variants were inserted into the same ImageNet-pretrained ResNet50 under a controlled protocol (identical split, preprocessing, augmentation, optimizer, schedule and classifier).

Self-attention at P3 gave the best accuracy (0.9437) and macro F1 (0.9435), followed by channel attention (0.9365), while spatial attention, ECA and CBAM fell below the baseline (0.9328). For Channel–Spatial attention, P3 was the best insertion position, and adding attention at several stages did not improve on a single well-placed module. Attention order and kernel size made no clear difference. Larger modules were not systematically better: the small channel module offered the best accuracy–cost trade-off, while self-attention gave the only larger gain at roughly 21% extra inference time.

Qualitatively, the learned spatial maps were nearly uniform and the feature maps changed little after attention, which is consistent with the lack of overall accuracy gain for Channel+Spatial. These conclusions carry a caveat: every model was trained once on a short 5-epoch schedule and tested on 551 images, so most differences lie within about ±2 points. Repeated runs with different seeds, the full-length schedule, and Grad-CAM taken at a matching layer for all models are needed to confirm the rankings.

Key takeaway: attention effectiveness depends on attention type, insertion position, feature resolution, feature semantics and model complexity together, so classification accuracy alone is not sufficient; it should be combined with computational cost and qualitative visualization.

## 8. References

1. He, K., Zhang, X., Ren, S. & Sun, J. — *Deep Residual Learning for Image Recognition* (CVPR 2016)
2. Hu, J., Shen, L., Albanie, S., Sun, G. & Wu, E. — *Squeeze-and-Excitation Networks* (TPAMI 2020)
3. Woo, S., Park, J., Lee, J.-Y. & Kweon, I. S. — *CBAM: Convolutional Block Attention Module* (ECCV 2018)
4. Wang, Q. et al. — *ECA-Net: Efficient Channel Attention for Deep CNNs* (CVPR 2020)
5. Hou, Q., Zhou, D. & Feng, J. — *Coordinate Attention for Efficient Mobile Network Design* (CVPR 2021)
6. Vaswani, A. et al. — *Attention Is All You Need* (NeurIPS 2017)
7. Selvaraju, R. R. et al. — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization* (ICCV 2017)
8. TensorFlow Documentation — https://www.tensorflow.org
9. Keras Documentation — https://keras.io
10. TensorFlow Datasets — https://www.tensorflow.org/datasets
