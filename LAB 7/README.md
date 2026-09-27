# CS3807 – Deep Learning Laboratory

## Experiment 7: End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders

**Institution:** Shiv Nadar University Chennai
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V
**Subject Code:** CS3807 – Deep Learning Laboratory
**Academic Year:** 2026–27

## 1. Objective

To develop an end-to-end understanding of autoencoders and their variants for image representation, reconstruction, denoising and generative modeling. The experiment covers:

- The encoder–latent–decoder structure of a fully connected autoencoder
- Convolutional Autoencoders (CAE) for spatially-aware reconstruction
- Denoising Autoencoders (DAE) trained against clean targets under Gaussian and salt-and-pepper corruption
- Variational Autoencoders (VAE): probabilistic latent representations and the reparameterization trick
- Quantitative evaluation using MSE, MAE and SSIM
- Latent-space visualization, interpolation and generative sampling

## 2. Dataset Description

### 2.1 MNIST Handwritten Digit Dataset

| Property | Value |
|---|---|
| Source | Keras/TensorFlow `mnist` dataset |
| Image dimensions | 28 × 28 × 1 (grayscale) |
| Training subset used | 10,000 images |
| Test subset used | 2,000 images |
| Number of classes | 10 (digits 0–9) |
| Pixel normalization | [0, 255] → [0, 1] |

Labels are not used as reconstruction targets — the original image itself is the target (`x → Encoder → z → Decoder → x̂`). The same normalized image set is reused across all four models (FC-AE, CAE, DAE, VAE) for a controlled comparison.

### 2.2 Noise Corruption (Denoising Autoencoder)

| Noise Type | Formulation | Levels Used |
|---|---|---|
| Gaussian | x̃ = x + n, n ~ N(0, σ²), clipped to [0, 1] | σ ∈ {0.1, 0.2, 0.3} |
| Salt-and-Pepper | random pixels set to 0 or 1 | p ∈ {0.05, 0.10, 0.20} |

## 3. Notebook Structure

| Section | Description |
|---|---|
| 0 – Setup & Dataset Preparation | Installs/imports TensorFlow-Keras, NumPy, Matplotlib, scikit-image; loads and normalizes MNIST; sets random seeds |
| 1 – Fully Connected Autoencoder (Plots 1–2) | `784 → 128 → 32 → 16(latent) → 32 → 128 → 784` architecture; trains with Adam + BCE loss; plots reconstructions and loss curves; computes MSE/MAE/SSIM |
| 2 – Convolutional Autoencoder (Plot 3) | Conv(32)→MaxPool→Conv(64)→MaxPool→Conv(64) encoder, UpSampling+Conv decoder; compares against FC-AE on reconstruction quality and parameter count |
| 3 – Denoising Convolutional Autoencoder (Plots 4–5) | Trains CAE against clean targets given Gaussian/salt-and-pepper corrupted inputs; evaluates reconstruction quality across noise levels |
| 4 – Variational Autoencoder (Plots 6–8) | 2-D latent space with reparameterization trick; visualizes latent space, samples new digits, and performs latent-space interpolation |
| 5 – Reconstruction Comparison Across Models | Consolidated MSE/MAE/SSIM/parameter/time table for FC-AE, CAE, DAE and VAE |
| 6 – Reconstruction Error Distribution (Plot 9) | Histogram of per-image MSE on the test set; identifies and displays the five hardest-to-reconstruct images |
| 7 – Additional Latent-Dimension Study (Plot 10) | Repeats the autoencoder with latent sizes 4/8/16/32 and reports the MSE/MAE/SSIM trade-off |
| 8 – Discussion (25 Questions) | Answers all 25 conceptual questions from the lab manual |
| 9 – Additional Exercises (7) | Latent-dimension sweep (CAE); Gaussian vs. salt-and-pepper comparison; DAE at multiple training noise levels; Conv2DTranspose vs. UpSampling2D decoder; VAE latent-dimension comparison (2 vs. 8); β-VAE KL-weight study; 100-sample VAE grid; centroid-to-centroid latent interpolation (digit 1 → 7) |

## 4. Results

### 4.1 Reconstruction Comparison Across Models

| Model | MSE | MAE | SSIM | Parameters | Time (s) |
|---|---|---|---|---|---|
| Fully Connected AE | 0.022507 | 0.058655 | 0.735063 | 211,040 | 18.79 |
| Convolutional AE | 0.002974 | 0.016280 | 0.968968 | 74,497 | 19.48 |
| Denoising CAE | 0.005036 | 0.022221 | 0.940905 | 74,497 | 16.57 |
| VAE | 0.044173 | 0.102887 | 0.483118 | 284,933 | 29.16 |

### 4.2 VAE Loss Components

| Metric | Value |
|---|---|
| Reconstruction Loss | 149.843140 |
| KL Loss | 6.049046 |
| Total Loss | 155.892181 |

Note: The VAE's reconstruction-only MSE/MAE/SSIM are reported separately because its training objective also includes the KL-divergence regularization term, which trades off pure reconstruction fidelity for a smooth, sample-able latent space.

### 4.3 Denoising Autoencoder — Noise Level vs. Quality (Gaussian)

| Noise Level (σ) | MSE | MAE | SSIM |
|---|---|---|---|
| 0.1 | 0.004042 | 0.018699 | 0.955667 |
| 0.2 | 0.005047 | 0.022242 | 0.940823 |
| 0.3 | 0.007544 | 0.030127 | 0.873935 |

### 4.4 Key Training Observations

| Model | Expected Behaviour |
|---|---|
| FC-AE Training/Validation Loss | Falls steeply over the first ~5 epochs then flattens near 0.125; curves stay closely aligned, indicating no strong overfitting |
| CAE vs FC-AE | CAE achieves lower MSE/MAE and higher SSIM with ~3× fewer parameters, since convolution preserves spatial locality and reuses weights across pixel neighbourhoods |
| Denoising CAE | Recovers major digit structure even under visible pixel corruption; degrades gradually rather than abruptly as noise level increases |
| VAE | Reconstruction is blurrier and less accurate than the deterministic CAE, but the latent space is continuous and supports meaningful sampling/interpolation |

### 4.5 Additional Exercise Summary

| Exercise | Finding |
|---|---|
| 1 – Latent-channel sweep (4/8/16/32, CAE) | MSE falls from 0.019347 → 0.004461 as channels increase; largest gains at low capacity, diminishing returns beyond 16 |
| 2 – Gaussian vs. Salt-and-Pepper noise | At matched nominal levels, salt-and-pepper degrades reconstruction more severely (e.g. SSIM 0.770 at p=0.20 vs. 0.874 at σ=0.30) since it replaces pixels with extreme values rather than perturbing them smoothly |
| 3 – DAE at two training noise levels | Training at σ=0.1 gives SSIM 0.9557; training at σ=0.3 gives SSIM 0.9020 — higher training noise trades off clean-input fidelity for robustness |
| 4 – Conv2DTranspose vs. UpSampling2D | Transposed convolutions substantially outperform UpSampling2D (MSE 0.000835 vs. 0.002974; SSIM 0.9902 vs. 0.9690) |
| 5 – VAE latent dimension (2 vs. 8) | Increasing latent dimension from 2 to 8 nearly halves MSE (0.046 → 0.024) and raises SSIM (0.454 → 0.734), at the cost of visualizability |
| 6 – β-VAE KL-weight sweep | Increasing β from 0.5 → 5.0 does not monotonically improve reconstruction; SSIM peaks near β=1.0 (0.4378), showing the reconstruction/regularization trade-off is not linear |
| 7 – 100-sample VAE grid & centroid interpolation | A 10×10 grid shows more diversity than a 5×5 grid but confirms the same class-overlap pattern seen in the 2-D latent scatter plot; centroid-to-centroid interpolation (digit 1 → 7) produces a smoother, more representative transition than interpolating between two arbitrary samples |

## 5. Mandatory Plots

| # | Description |
|---|---|
| 1 | Original versus reconstructed images (FC-AE) |
| 2 | Training and validation reconstruction loss (FC-AE) |
| 3 | Fully connected AE versus convolutional AE reconstruction comparison |
| 4 | Clean versus noisy versus denoised images |
| 5 | Noise level versus MSE / MAE / SSIM |
| 6 | VAE latent-space visualization (2-D scatter, coloured by digit label) |
| 7 | Randomly generated VAE images |
| 8 | Latent-space interpolation |
| 9 | Distribution of per-image reconstruction errors on the test set |
| 10 | Latent dimension versus reconstruction MSE |

Each plot is followed by an inline inference covering: what the plot shows, the trend observed, why it occurs, and the conclusion drawn.

## 6. Conceptual Highlights

**Autoencoder Fundamentals**
An autoencoder learns `z = f_θ(x)` (encoder) and `x̂ = g_φ(z)` (decoder), trained to minimize reconstruction error `L_rec = (1/N)Σ‖x_i − x̂_i‖²`. The bottleneck forces the network to learn a compact, informative representation rather than a trivial identity mapping.

**Fully Connected vs. Convolutional**
The FC-AE flattens each image into an unordered 784-length vector, discarding explicit spatial relationships. The CAE instead applies convolution and pooling directly to the 28×28×1 image, using weight sharing to learn reusable stroke-like filters — achieving lower reconstruction error with far fewer parameters.

**Denoising Autoencoder**
A denoising autoencoder receives a corrupted input `x̃ → Encoder → z → Decoder → x̂` but is always trained against the clean image `x` as target. This forces the bottleneck to act as a structural filter that suppresses unpredictable noise while retaining the digit's consistent, learnable structure.

**Variational Autoencoder**
Rather than a deterministic `z = f(x)`, a VAE learns a distribution `q_φ(z|x) = N(μ(x), diag(σ²(x)))`. The reparameterization trick `z = μ + σ ⊙ ε, ε ~ N(0, I)` allows gradients to flow through the sampling step. The loss combines a reconstruction term with a KL-divergence term `D_KL(q_φ(z|x) ‖ p(z))` that regularizes the latent space toward a standard normal prior, making it continuous and suitable for sampling and interpolation.

## 7. Conclusion

All four models — Fully Connected Autoencoder, Convolutional Autoencoder, Denoising Convolutional Autoencoder, and Variational Autoencoder — were implemented, trained, and evaluated on the MNIST dataset under a controlled experimental protocol.

The Convolutional Autoencoder substantially outperforms the Fully Connected Autoencoder on every reconstruction metric while using roughly a third of the parameters, confirming that preserving spatial locality through convolution is far more parameter-efficient than flattening. The Denoising Autoencoder demonstrates that the same convolutional bottleneck can learn a noise-robust mapping from corrupted inputs to clean targets, with reconstruction quality degrading gradually — rather than catastrophically — as corruption increases. The Variational Autoencoder trades some reconstruction fidelity for a smooth, continuous, samplable latent space: its 2-D latent scatter shows partial class clustering with expected overlap, its randomly generated samples are mostly recognizable digits, and its interpolations (both between individual samples and between class centroids) transition smoothly rather than jumping abruptly between digit shapes.

Key takeaway: deterministic autoencoders (FC-AE, CAE, DAE) optimize purely for reconstruction accuracy, while the VAE optimizes a trade-off between reconstruction and a well-structured, generative latent space — the right choice depends on whether the downstream goal is faithful reconstruction or novel sample generation.

## 8. References

1. Goodfellow, I., Bengio, Y. & Courville, A. — *Deep Learning* (MIT Press, 2016)
2. Kingma, D. P. & Welling, M. — *Auto-Encoding Variational Bayes* (ICLR 2014)
3. Vincent, P. et al. — *Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion* (JMLR 2010)
4. LeCun, Y., Cortes, C. & Burges, C. J. C. — MNIST Handwritten Digit Database
5. TensorFlow Documentation — https://www.tensorflow.org
6. Keras Documentation — https://keras.io
7. scikit-image Documentation — https://scikit-image.org
