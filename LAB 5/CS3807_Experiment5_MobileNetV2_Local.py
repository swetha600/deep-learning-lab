"""
CS3807 - Deep Learning Laboratory
Experiment 5: MobileNetV2 on the Oxford-IIIT Pet Dataset
SINGLE FILE, FOR LOCAL EXECUTION (not Colab).

Covers the full lab manual: Sections 0, 3-17.

======================================================================
 HOW TO RUN THIS LOCALLY
======================================================================
1) Create and activate a virtual environment (recommended):
       python3 -m venv venv
       source venv/bin/activate        # Windows: venv\\Scripts\\activate

2) Install dependencies:
       pip install --upgrade pip
       pip install tensorflow tensorflow-datasets pandas matplotlib scikit-learn

   GPU notes (optional but much faster than CPU):
     - NVIDIA GPU + recent TensorFlow (2.16+): the regular `tensorflow` pip
       package now bundles CUDA support on Linux -- just make sure your
       NVIDIA driver is up to date. Verify with:
           python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
     - Apple Silicon (M1/M2/M3): use `pip install tensorflow-macos tensorflow-metal` instead.
     - No GPU: this will still run on CPU, just noticeably slower --
       consider reducing the EPOCHS_* constants below if it's too slow.

3) Run it:
       python CS3807_Experiment5_MobileNetV2_Local.py

   The first run downloads the Oxford-IIIT Pet dataset (~800MB) via
   tensorflow_datasets into ~/tensorflow_datasets -- it's cached there for
   all future runs, including on other scripts/notebooks on this machine.

======================================================================
 IF YOUR MACHINE CRASHES, FREEZES, OR YOU NEED TO STOP MID-RUN
======================================================================
Every long-running section below (weight init, regularization, optimizers,
hyperparameter sweeps, K-fold CV, the additional exercise) checkpoints its
progress to a local `checkpoints/` folder after EVERY individual training
run (not just at the end of a whole section). If you stop the script
(Ctrl+C) or it crashes and you re-run it, it will:
    - skip anything already checkpointed, and
    - resume from wherever it left off,
so you never lose more than the single training run that was interrupted.
Delete a specific .pkl file (or the whole `checkpoints/` folder) if you
ever want to force a clean re-run of that part.

Since this runs as a plain script, "crash" here usually means: out of RAM
(see the note further down), your machine went to sleep, or you closed the
terminal. If you're on a laptop, disable sleep/screen-lock-triggered
suspension while this runs, or run it inside `tmux`/`screen` (Linux/macOS)
so it survives a disconnected terminal.

If you hit an actual out-of-memory crash (the process gets killed, or you
see "Killed" in the terminal on Linux): lower EPOCHS_CV / K in Section 11,
lower the batch sizes tried in Section 9, or close other memory-hungry
applications before running. This script already calls
tf.keras.backend.clear_session() + gc.collect() between every training run
so memory from one run doesn't carry into the next -- most local crashes
come from having too many other things open, or from a machine with very
limited RAM (<8GB) trying to hold ~4GB of preprocessed images in memory at
once (Section 11 onward). See the comment above `dataset_to_arrays` if
that's the bottleneck for you.
"""

import os
import io
import time
import math
import random
import gc
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# ==========================================================================
# 0. Setup and Imports
# ==========================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices("GPU"))

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 11,
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.eps"), format="eps", bbox_inches="tight")
    plt.close(fig)


# --- Checkpointing helpers (this is what makes crashes non-fatal) ---------
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)


def load_ckpt(name):
    path = os.path.join(CKPT_DIR, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_ckpt(name, data):
    path = os.path.join(CKPT_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ==========================================================================
# 3. Dataset and Experimental Setup
# ==========================================================================
IMG_SIZE = (224, 224)
NUM_CLASSES = 37
BATCH_SIZE_DEFAULT = 32
AUTOTUNE = tf.data.AUTOTUNE

(raw_train_full, raw_test), ds_info = tfds.load(
    "oxford_iiit_pet",
    split=["train", "test"],
    with_info=True,
    as_supervised=True,
)

print(ds_info)
CLASS_NAMES = ds_info.features["label"].names
print("Number of classes:", len(CLASS_NAMES))


def preprocess(image, label, augment=False):
    image = tf.image.resize(image, IMG_SIZE)
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def make_dataset(ds, batch_size=BATCH_SIZE_DEFAULT, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1024, seed=SEED)
    ds = ds.map(lambda x, y: preprocess(x, y, augment=augment), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


n_train_full = int(ds_info.splits["train"].num_examples)
raw_train_full = raw_train_full.shuffle(n_train_full, seed=SEED, reshuffle_each_iteration=False)
n_val = int(0.2 * n_train_full)

raw_val = raw_train_full.take(n_val)
raw_train = raw_train_full.skip(n_val)

print(
    f"Train examples: {n_train_full - n_val} | Val examples: {n_val} "
    f"| Test examples: {ds_info.splits['test'].num_examples}"
)

train_ds = make_dataset(raw_train, shuffle=True, augment=True)
val_ds = make_dataset(raw_val)
test_ds = make_dataset(raw_test)


# ==========================================================================
# 4. MobileNetV2 Architecture - Model Builder
# ==========================================================================
def build_model(
    init="glorot_uniform",
    dropout_rate=0.0,
    use_batchnorm=True,
    l2_reg=0.0,
    trainable_base=False,
    fine_tune_at=None,
    optimizer="adam",
    learning_rate=1e-3,
):
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base.trainable = bool(trainable_base)
    if trainable_base and fine_tune_at is not None:
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False if not trainable_base else None)
    x = layers.GlobalAveragePooling2D()(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu", kernel_initializer=init, kernel_regularizer=reg)(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", kernel_initializer=init, kernel_regularizer=reg)(x)

    model = keras.Model(inputs, outputs)

    opt_map = {
        "sgd": optimizers.SGD(learning_rate=learning_rate),
        "momentum": optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        "rmsprop": optimizers.RMSprop(learning_rate=learning_rate),
        "adam": optimizers.Adam(learning_rate=learning_rate),
    }
    opt = opt_map[optimizer] if isinstance(optimizer, str) else optimizer

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base


print("build_model() ready.")


def run_config_sweep(configs, ckpt_name, train_one_fn):
    """Generic resumable sweep: trains whichever keys of `configs` are not
    already present in the checkpoint, saving after every single one."""
    results = load_ckpt(ckpt_name)
    print(f"[{ckpt_name}] Resuming: {len(results)}/{len(configs)} already done -> {list(results.keys())}")
    for name, cfg in configs.items():
        if name in results:
            print(f"[{ckpt_name}] '{name}' already done -- skipping.")
            continue
        print(f"[{ckpt_name}] --- Training '{name}' ---")
        tf.keras.backend.clear_session()
        gc.collect()
        result = train_one_fn(name, cfg)
        gc.collect()
        results[name] = result
        save_ckpt(ckpt_name, results)
    return results


# ==========================================================================
# 5. Weight Initialization
# ==========================================================================
INIT_CONFIGS = {
    "Zero": "zeros",
    "Random": keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=SEED),
    "Xavier/Glorot": "glorot_uniform",
    "He": "he_normal",
}
EPOCHS_INIT = 10


def _train_init(name, init):
    model, _ = build_model(init=init, dropout_rate=0.0, use_batchnorm=False, trainable_base=False)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_INIT, verbose=1)
    result = hist.history
    del model
    return result


init_histories = run_config_sweep(INIT_CONFIGS, "init_histories", _train_init)

fig, ax = plt.subplots(figsize=(7, 5))
for name, h in init_histories.items():
    ax.plot(range(1, len(h["loss"]) + 1), h["loss"], marker="o", label=name)
ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
ax.set_title("Plot 1: Training Loss vs. Epoch - Weight Initialization")
ax.legend(); ax.grid(alpha=0.3)
save_fig(fig, "plot01_init_train_loss")

fig, ax = plt.subplots(figsize=(7, 5))
for name, h in init_histories.items():
    ax.plot(range(1, len(h["val_accuracy"]) + 1), [a * 100 for a in h["val_accuracy"]], marker="o", label=name)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("Plot 2: Validation Accuracy vs. Epoch - Weight Initialization")
ax.legend(); ax.grid(alpha=0.3)
save_fig(fig, "plot02_init_val_acc")

# Inference (Plots 1 & 2) -- fill in the [...] with the numbers your run produces:
# 1. What it shows: the effect of Zero, Random, Xavier and He initialization on
#    how quickly the classifier head's training loss falls and its validation
#    accuracy rises.
# 2. Trend observed: Zero init is expected to stagnate ([...]% val. accuracy)
#    because every neuron in a layer receives identical gradients and never
#    symmetry-breaks; Xavier and He should converge fastest and reach the
#    highest validation accuracy ([...]% vs [...]%), with Random in between.
# 3. Why: Xavier/He scale initial weight variance to the layer's fan-in/out,
#    keeping activations/gradients well-behaved at the start of training.


# ==========================================================================
# 6. Regularization and Overfitting
# ==========================================================================
REG_CONFIGS = {
    "No Regularization": dict(dropout_rate=0.0, use_batchnorm=False, l2_reg=0.0),
    "L2 Regularization": dict(dropout_rate=0.0, use_batchnorm=False, l2_reg=1e-3),
    "Dropout": dict(dropout_rate=0.5, use_batchnorm=False, l2_reg=0.0),
    "Batch Normalization": dict(dropout_rate=0.0, use_batchnorm=True, l2_reg=0.0),
}
EPOCHS_REG = 12


def _train_reg(name, cfg):
    model, _ = build_model(init="he_normal", trainable_base=False, **cfg)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_REG, verbose=1)
    result = hist.history
    del model
    return result


reg_histories = run_config_sweep(REG_CONFIGS, "reg_histories", _train_reg)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, (name, h) in zip(axes.ravel(), reg_histories.items()):
    ax.plot(h["accuracy"], marker="o", label="Train Acc")
    ax.plot(h["val_accuracy"], marker="s", label="Val Acc")
    ax.set_title(name); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("Plot 3: Training and Validation Accuracy vs. Epoch - Regularization")
fig.tight_layout()
save_fig(fig, "plot03_reg_accuracy")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, (name, h) in zip(axes.ravel(), reg_histories.items()):
    ax.plot(h["loss"], marker="o", label="Train Loss")
    ax.plot(h["val_loss"], marker="s", label="Val Loss")
    ax.set_title(name); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("Plot 4: Training and Validation Loss vs. Epoch - Regularization")
fig.tight_layout()
save_fig(fig, "plot04_reg_loss")

# Inference (Plots 3 & 4): fill in [...] with your run's numbers.
# 1. What it shows: the train/val accuracy & loss gap for each strategy.
# 2. Trend observed: "No Regularization" should show the largest gap
#    (train acc -> [...]%,  val acc plateaus near [...]%); Dropout/L2/BN
#    narrow it.
# 3. Why: Dropout/L2 limit the head's capacity to memorise; BN speeds/
#    stabilises convergence.


# ==========================================================================
# 7. Batch Normalization
# ==========================================================================
x = np.array([2.0, 4.0, 6.0, 8.0])
eps = 1e-8
mu_B = x.mean()
var_B = ((x - mu_B) ** 2).mean()
x_hat = (x - mu_B) / np.sqrt(var_B + eps)
gamma, beta = 1.0, 0.0
y = gamma * x_hat + beta

print(f"mu_B  = {mu_B}")
print(f"var_B = {var_B}")
print(f"x_hat = {np.round(x_hat, 3)}")
print(f"y (gamma=1, beta=0) = {np.round(y, 3)}")

bn_layer = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=eps)
bn_out = bn_layer(x.reshape(1, 4, 1).astype("float32"), training=True)
print("TF BatchNormalization output:", bn_out.numpy().ravel())

# Plot 5: With vs. Without Batch Normalization
h_with_bn = reg_histories["Batch Normalization"]
h_without_bn = reg_histories["No Regularization"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([a * 100 for a in h_with_bn["val_accuracy"]], marker="o", label="With BN")
ax.plot([a * 100 for a in h_without_bn["val_accuracy"]], marker="s", label="Without BN")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("Plot 5: With vs. Without Batch Normalization")
ax.legend(); ax.grid(alpha=0.3)
save_fig(fig, "plot05_bn_comparison")

# Inference: BN re-centres/rescales activations each batch, reducing internal
# covariate shift; expect faster, more stable early convergence ([...]% vs
# [...]% by epoch 3-4, say).


# ==========================================================================
# 8. Optimization Algorithms
# ==========================================================================
OPTIMIZER_CONFIGS = {name: name for name in ["sgd", "momentum", "rmsprop", "adam"]}
EPOCHS_OPT = 12
LR_OPT = 1e-3
_opt_times = load_ckpt("opt_times")


def _train_opt(name, opt_name):
    model, _ = build_model(init="he_normal", trainable_base=False, optimizer=opt_name, learning_rate=LR_OPT)
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_OPT, verbose=1)
    _opt_times[name] = time.time() - t0
    save_ckpt("opt_times", _opt_times)
    result = hist.history
    del model
    return result


opt_histories = run_config_sweep(OPTIMIZER_CONFIGS, "opt_histories", _train_opt)
opt_times = load_ckpt("opt_times")

fig, ax = plt.subplots(figsize=(7, 5))
for name, h in opt_histories.items():
    ax.plot(h["loss"], marker="o", label=name.upper())
ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
ax.set_title("Plot 6: Training Loss vs. Epoch - Optimizers")
ax.legend(); ax.grid(alpha=0.3)
save_fig(fig, "plot06_optimizer_train_loss")

fig, ax = plt.subplots(figsize=(7, 5))
for name, h in opt_histories.items():
    ax.plot([a * 100 for a in h["val_accuracy"]], marker="o", label=name.upper())
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("Plot 7: Validation Accuracy vs. Epoch - Optimizers")
ax.legend(); ax.grid(alpha=0.3)
save_fig(fig, "plot07_optimizer_val_acc")

rows = []
for name, h in opt_histories.items():
    best_epoch = int(np.argmax(h["val_accuracy"])) + 1
    rows.append({
        "Optimizer": name.upper(),
        "Final Loss": round(h["loss"][-1], 4),
        "Best Val. Accuracy (%)": round(max(h["val_accuracy"]) * 100, 2),
        "Epoch to Converge": best_epoch,
        "Time (s)": round(opt_times.get(name, float("nan")), 1),
    })
optimizer_table = pd.DataFrame(rows)
print(optimizer_table)

# Inference: SGD slowest/noisiest; Momentum smooths it; RMSProp/Adam
# (adaptive per-parameter LR) converge fastest, typically highest val acc.


# ==========================================================================
# 9. CNN Hyperparameter Tuning
# ==========================================================================
BASELINE = dict(optimizer="adam", learning_rate=1e-3, dropout_rate=0.25, use_batchnorm=False, init="he_normal", trainable_base=False)
EPOCHS_HP = 8


def eval_val_accuracy(**overrides):
    cfg = {**BASELINE, **overrides}
    batch_size = cfg.pop("batch_size", BATCH_SIZE_DEFAULT)
    tf.keras.backend.clear_session()
    gc.collect()
    model, _ = build_model(**cfg)
    tr = make_dataset(raw_train, batch_size=batch_size, shuffle=True, augment=True)
    va = make_dataset(raw_val, batch_size=batch_size)
    hist = model.fit(tr, validation_data=va, epochs=EPOCHS_HP, verbose=0)
    val_acc = max(hist.history["val_accuracy"]) * 100
    del model
    gc.collect()
    return val_acc


LR_VALUES = [0.001, 0.0001]
BS_VALUES = [16, 32, 64]
DO_VALUES = [0.0, 0.25, 0.5]

hp_ckpt = load_ckpt("hp_results")
lr_results = hp_ckpt.get("lr", {})
bs_results = hp_ckpt.get("bs", {})
do_results = hp_ckpt.get("do", {})


def _save_hp():
    save_ckpt("hp_results", {"lr": lr_results, "bs": bs_results, "do": do_results})


for lr in LR_VALUES:
    if lr in lr_results:
        print(f"lr={lr} already evaluated -- skipping.")
        continue
    lr_results[lr] = eval_val_accuracy(learning_rate=lr)
    print(f"lr={lr} -> val_accuracy={lr_results[lr]:.2f}%")
    _save_hp()

for bs in BS_VALUES:
    if bs in bs_results:
        print(f"batch_size={bs} already evaluated -- skipping.")
        continue
    bs_results[bs] = eval_val_accuracy(batch_size=bs)
    print(f"batch_size={bs} -> val_accuracy={bs_results[bs]:.2f}%")
    _save_hp()

for do in DO_VALUES:
    if do in do_results:
        print(f"dropout_rate={do} already evaluated -- skipping.")
        continue
    do_results[do] = eval_val_accuracy(dropout_rate=do)
    print(f"dropout_rate={do} -> val_accuracy={do_results[do]:.2f}%")
    _save_hp()

print("Learning rate results:", lr_results)
print("Batch size results:", bs_results)
print("Dropout rate results:", do_results)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([str(k) for k in lr_results.keys()], list(lr_results.values()), marker="o")
ax.set_xlabel("Learning Rate"); ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("Plot 8: Learning Rate vs. Validation Accuracy")
ax.grid(alpha=0.3)
save_fig(fig, "plot08_lr_vs_acc")

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([str(k) for k in bs_results.keys()], list(bs_results.values()), marker="o", color="tab:orange")
ax.set_xlabel("Batch Size"); ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("Plot 9: Batch Size vs. Validation Accuracy")
ax.grid(alpha=0.3)
save_fig(fig, "plot09_batchsize_vs_acc")

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([str(k) for k in do_results.keys()], list(do_results.values()), marker="o", color="tab:green")
ax.set_xlabel("Dropout Rate"); ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("Plot 10: Dropout Rate vs. Validation Accuracy")
ax.grid(alpha=0.3)
save_fig(fig, "plot10_dropout_vs_acc")

# Inference: 1e-3 should outperform 1e-4 within this short budget; batch
# size effect smaller/noisier; dropout ~0.25-0.5 should beat 0.


# ==========================================================================
# 10. Transfer Learning and Fine-Tuning
# ==========================================================================
EPOCHS_TL = 10
tl_ckpt = load_ckpt("transfer_learning")


class _Hist:
    def __init__(self, h):
        self.history = h


if "feature_extraction" in tl_ckpt:
    print("Feature extraction already trained -- skipping.")
    hist_fe = _Hist(tl_ckpt["feature_extraction"])
else:
    tf.keras.backend.clear_session()
    gc.collect()
    model_fe, base_fe = build_model(init="he_normal", dropout_rate=0.25, trainable_base=False, optimizer="adam", learning_rate=1e-3)
    hist_fe = model_fe.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_TL, verbose=1)
    print(f"Total base layers: {len(base_fe.layers)}")
    tl_ckpt["feature_extraction"] = hist_fe.history
    save_ckpt("transfer_learning", tl_ckpt)
    del model_fe, base_fe
    gc.collect()

FINE_TUNE_AT = 100
if "fine_tuning" in tl_ckpt:
    print("Fine-tuning already trained -- skipping.")
    hist_ft = _Hist(tl_ckpt["fine_tuning"])
else:
    tf.keras.backend.clear_session()
    gc.collect()
    model_ft, base_ft = build_model(
        init="he_normal", dropout_rate=0.25,
        trainable_base=True, fine_tune_at=FINE_TUNE_AT,
        optimizer="adam", learning_rate=1e-5,
    )
    hist_ft = model_ft.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_TL, verbose=1)
    tl_ckpt["fine_tuning"] = hist_ft.history
    save_ckpt("transfer_learning", tl_ckpt)
    del model_ft, base_ft
    gc.collect()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([a * 100 for a in hist_fe.history["val_accuracy"]], marker="o", label="Feature Extraction")
ax.plot([a * 100 for a in hist_ft.history["val_accuracy"]], marker="s", label="Fine-Tuning")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("Plot 11: Feature Extraction vs. Fine-Tuning")
ax.legend(); ax.grid(alpha=0.3)
save_fig(fig, "plot11_fe_vs_ft")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axes[0].plot(hist_fe.history["loss"], marker="o", label="Train Loss")
axes[0].plot(hist_fe.history["val_loss"], marker="s", label="Val Loss")
axes[0].set_title("Feature Extraction"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(hist_ft.history["loss"], marker="o", label="Train Loss")
axes[1].plot(hist_ft.history["val_loss"], marker="s", label="Val Loss")
axes[1].set_title("Fine-Tuning"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle("Plot 12: Training and Validation Loss - Before vs. After Fine-Tuning")
fig.tight_layout()
save_fig(fig, "plot12_finetune_loss")

# Inference: fine-tuning should edge out feature extraction once the
# unfrozen upper layers adapt ([...]% vs [...]%). Smaller LR during
# fine-tuning avoids destroying the pretrained weights.


# ==========================================================================
# 11. K-Fold Cross-Validation
# ==========================================================================
# NOTE ON MEMORY: this materializes the full training pool as a numpy array
# (a few GB). If your machine has limited RAM (<8GB free), this is the most
# likely crash point. It's cached to disk after the first run so you never
# redo this step. If it's still too much, edit `dataset_to_arrays` below to
# subsample raw_train_full (e.g. `.take(3000)`) before converting, at the
# cost of a less representative cross-validation.
def dataset_to_arrays(raw_ds):
    images, labels = [], []
    for img, lbl in raw_ds:
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        images.append(img.numpy())
        labels.append(lbl.numpy())
    return np.array(images, dtype="float32"), np.array(labels)


X_PATH = os.path.join(CKPT_DIR, "X_train_full.npy")
Y_PATH = os.path.join(CKPT_DIR, "y_train_full.npy")

if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
    print("Loading cached X_train_full / y_train_full from disk.")
    X_train_full = np.load(X_PATH)
    y_train_full = np.load(Y_PATH)
else:
    X_train_full, y_train_full = dataset_to_arrays(raw_train_full)
    np.save(X_PATH, X_train_full)
    np.save(Y_PATH, y_train_full)
print(X_train_full.shape, y_train_full.shape)

CANDIDATE_CONFIGS = {
    "C1": dict(init="he_normal", dropout_rate=0.25, use_batchnorm=False, l2_reg=0.0, optimizer="adam", learning_rate=1e-3, trainable_base=False),
    "C2": dict(init="he_normal", dropout_rate=0.5, use_batchnorm=True, l2_reg=0.0, optimizer="adam", learning_rate=1e-3, trainable_base=False),
    "C3": dict(init="glorot_uniform", dropout_rate=0.25, use_batchnorm=False, l2_reg=1e-3, optimizer="rmsprop", learning_rate=1e-3, trainable_base=False),
    "C4": dict(init="he_normal", dropout_rate=0.25, use_batchnorm=False, l2_reg=0.0, optimizer="adam", learning_rate=1e-5, trainable_base=True, fine_tune_at=100),
}
K = 5
EPOCHS_CV = 6
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)


def run_cv_folds(name, cfg, batch_size=32, results_dict=None, ckpt_name="cv_results"):
    """Resumable, fold-by-fold: at most one fold's work is ever lost."""
    accs = results_dict[name]
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_full), start=1):
        if fold <= len(accs):
            print(f"  [{name}] Fold {fold} already done ({accs[fold - 1]:.2f}%) -- skipping.")
            continue
        tf.keras.backend.clear_session()
        gc.collect()
        y_tr = tf.one_hot(y_train_full[tr_idx], NUM_CLASSES).numpy()
        y_va = tf.one_hot(y_train_full[va_idx], NUM_CLASSES).numpy()
        model, _ = build_model(**cfg)
        model.fit(X_train_full[tr_idx], y_tr, validation_data=(X_train_full[va_idx], y_va),
                  epochs=EPOCHS_CV, batch_size=batch_size, verbose=0)
        val_acc = model.evaluate(X_train_full[va_idx], y_va, verbose=0)[1] * 100
        accs.append(val_acc)
        print(f"  [{name}] Fold {fold}: {val_acc:.2f}%")
        del model, y_tr, y_va
        gc.collect()
        save_ckpt(ckpt_name, results_dict)


cv_results = load_ckpt("cv_results")
for name in CANDIDATE_CONFIGS:
    cv_results.setdefault(name, [])

for name, cfg in CANDIDATE_CONFIGS.items():
    print(f"=== Cross-validating configuration {name} ===")
    run_cv_folds(name, cfg, results_dict=cv_results, ckpt_name="cv_results")

cv_rows = []
for name, accs in cv_results.items():
    row = {"Configuration": name}
    row.update({f"F{i+1}": round(a, 2) for i, a in enumerate(accs)})
    row["Mean +/- SD"] = f"{np.mean(accs):.2f} +/- {np.std(accs):.2f}"
    cv_rows.append(row)
cv_table = pd.DataFrame(cv_rows)
print(cv_table)

means = [np.mean(cv_results[c]) for c in CANDIDATE_CONFIGS]
sds = [np.std(cv_results[c]) for c in CANDIDATE_CONFIGS]
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(list(CANDIDATE_CONFIGS.keys()), means, yerr=sds, capsize=6, color="tab:blue", alpha=0.8)
ax.set_xlabel("Hyperparameter Configuration"); ax.set_ylabel("Mean Validation Accuracy (%)")
ax.set_title("Plot 13: 5-Fold Cross-Validation Accuracy (+/- SD)")
ax.grid(alpha=0.3, axis="y")
save_fig(fig, "plot13_cv_accuracy")

best_config_name = max(CANDIDATE_CONFIGS, key=lambda c: np.mean(cv_results[c]))
print(f"Best configuration by mean CV accuracy: {best_config_name}")


# ==========================================================================
# 12. Final Model Evaluation
# ==========================================================================
best_cfg = CANDIDATE_CONFIGS[best_config_name]
EPOCHS_FINAL = 15

final_train_ds = make_dataset(raw_train_full, shuffle=True, augment=True)
final_test_ds = make_dataset(raw_test)

FINAL_MODEL_WEIGHTS = os.path.join(CKPT_DIR, "final_model.weights.h5")
final_meta = load_ckpt("final_model_meta")

tf.keras.backend.clear_session()
gc.collect()
final_model, _ = build_model(**best_cfg)

if os.path.exists(FINAL_MODEL_WEIGHTS) and "final_train_time" in final_meta:
    print("Final model already trained -- loading saved weights.")
    final_model.load_weights(FINAL_MODEL_WEIGHTS)
    final_train_time = final_meta["final_train_time"]
else:
    t0 = time.time()
    final_history = final_model.fit(final_train_ds, epochs=EPOCHS_FINAL, verbose=1)
    final_train_time = time.time() - t0
    final_model.save_weights(FINAL_MODEL_WEIGHTS)
    save_ckpt("final_model_meta", {"final_train_time": final_train_time})

test_loss, test_acc = final_model.evaluate(final_test_ds, verbose=1)
n_params = final_model.count_params()
print(f"Test accuracy: {test_acc*100:.2f}% | Training time: {final_train_time:.1f}s | Params: {n_params:,}")

y_true, y_pred = [], []
for imgs, labels in final_test_ds:
    preds = final_model.predict(imgs, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
y_true, y_pred = np.array(y_true), np.array(y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

final_metrics_table = pd.DataFrame([{
    "Mean CV Accuracy (%)": round(np.mean(cv_results[best_config_name]), 2),
    "CV Standard Deviation": round(np.std(cv_results[best_config_name]), 2),
    "Test Accuracy (%)": round(test_acc * 100, 2),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1-score": round(f1, 4),
    "Training Time (s)": round(final_train_time, 1),
    "Number of Parameters": n_params,
}]).T.rename(columns={0: "Value"})
print(final_metrics_table)

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
ax.set_title("Plot 14: Confusion Matrix - 37 Pet Breeds")
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=6)
ax.set_yticklabels(CLASS_NAMES, fontsize=6)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
save_fig(fig, "plot14_confusion_matrix")

cm_no_diag = cm.copy().astype(float)
np.fill_diagonal(cm_no_diag, 0)
top_confusions = np.dstack(np.unravel_index(np.argsort(-cm_no_diag.ravel())[:5], cm_no_diag.shape))[0]
print("Top confused class pairs (true -> predicted, count):")
for t, p in top_confusions:
    print(f"  {CLASS_NAMES[t]} -> {CLASS_NAMES[p]}: {int(cm[t, p])}")

# Optional Plot 15: Misclassified Images (only fetches the ~9 needed images)
misclassified_idx = np.where(y_true != y_pred)[0]
sample_idx = np.random.choice(misclassified_idx, size=min(9, len(misclassified_idx)), replace=False)
sample_idx_set = set(sample_idx.tolist())

sample_images = {}
for i, (img, lbl) in enumerate(raw_test):
    if i in sample_idx_set:
        sample_images[i] = tf.image.resize(img, IMG_SIZE).numpy().astype("uint8")
        if len(sample_images) == len(sample_idx_set):
            break

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, idx in zip(axes.ravel(), sample_idx):
    ax.imshow(sample_images[idx])
    ax.set_title(f"True: {CLASS_NAMES[y_true[idx]]}\nPred: {CLASS_NAMES[y_pred[idx]]}", fontsize=8)
    ax.axis("off")
fig.suptitle("Optional Plot 15: Representative Misclassified Images")
fig.tight_layout()
save_fig(fig, "plot15_misclassified")


# ==========================================================================
# 13. Overall Results
# ==========================================================================
overall_rows = [
    {"Configuration": "Baseline (No Reg., Adam, He init, frozen base)",
     "CV Accuracy": "-", "SD": "-",
     "Test Accuracy": f"{max(reg_histories['No Regularization']['val_accuracy'])*100:.2f}%",
     "Training Time": "-"},
    {"Configuration": f"Best Initialization ({max(init_histories, key=lambda k: max(init_histories[k]['val_accuracy']))})",
     "CV Accuracy": "-", "SD": "-",
     "Test Accuracy": f"{max(max(h['val_accuracy']) for h in init_histories.values())*100:.2f}%",
     "Training Time": "-"},
    {"Configuration": f"Best Regularization ({max(reg_histories, key=lambda k: max(reg_histories[k]['val_accuracy']))})",
     "CV Accuracy": "-", "SD": "-",
     "Test Accuracy": f"{max(max(h['val_accuracy']) for h in reg_histories.values())*100:.2f}%",
     "Training Time": "-"},
    {"Configuration": f"Best Optimizer ({max(opt_histories, key=lambda k: max(opt_histories[k]['val_accuracy'])).upper()})",
     "CV Accuracy": "-", "SD": "-",
     "Test Accuracy": f"{max(max(h['val_accuracy']) for h in opt_histories.values())*100:.2f}%",
     "Training Time": "-"},
    {"Configuration": "Best Hyperparameters (Sec. 9 sweep)",
     "CV Accuracy": "-", "SD": "-",
     "Test Accuracy": f"{max(list(lr_results.values()) + list(bs_results.values()) + list(do_results.values())):.2f}%",
     "Training Time": "-"},
    {"Configuration": f"Fine-Tuned Model / Final Selected ({best_config_name})",
     "CV Accuracy": f"{np.mean(cv_results[best_config_name]):.2f}%",
     "SD": f"{np.std(cv_results[best_config_name]):.2f}",
     "Test Accuracy": f"{test_acc*100:.2f}%",
     "Training Time": f"{final_train_time:.1f}s"},
]
overall_results_table = pd.DataFrame(overall_rows)
print(overall_results_table)

# Justification (Section 13): using your run's actual numbers, state why the
# selected configuration was chosen -- highest mean CV accuracy ([...]%)
# with acceptable SD ([...]), confirmed on the held-out test set ([...]%).


# ==========================================================================
# 14. Required Inference for Plots
# ==========================================================================
# Per the manual, a short inference (what it shows / trend / why) is placed
# as a comment under each plot above (Plots 1-15). Fill in the bracketed
# [...] placeholders with your actual run's numbers before submitting.


# ==========================================================================
# 15. Discussion Questions
# ==========================================================================
# 1. Parameters vs. hyperparameters: parameters (weights/biases, BN's
#    gamma/beta) are learned from data; hyperparameters (LR, batch size,
#    dropout, optimizer) are set before training.
# 2. Weight init matters because it sets where optimization starts --
#    affects convergence speed/stability.
# 3. Zero init is problematic: every neuron in a layer gets identical
#    gradients (symmetry never breaks).
# 4. Xavier vs. He: Xavier scales variance by fan-in/fan-out for
#    linear/symmetric activations; He uses 2/fan-in, suited to ReLU.
# 5. Overfitting shows as a widening train/val accuracy or loss gap.
# 6. Dropout reduces overfitting by randomly zeroing activations, preventing
#    co-adaptation (acts like an implicit ensemble).
# 7. Batch Normalization normalises per-batch activations (mean 0, var 1)
#    before a learnable scale/shift, reducing internal covariate shift.
# 8. Numerical BN example: x=[2,4,6,8] -> mean=5, var=5,
#    x_hat ~= [-1.342, -0.447, 0.447, 1.342] (verified above against Keras).
# 9. Gamma/beta let BN learn to undo/adjust the normalisation per channel.
# 10. SGD (fixed step) < Momentum (damps oscillation) < RMSProp/Adam
#     (adaptive per-parameter learning rates).
# 11. Too-large LR: overshoot/divergence/oscillation.
# 12. Too-small LR: very slow convergence, can stall on plateaus.
# 13. Larger batch size: less noisy gradient, better hardware use, but less
#     beneficial stochastic noise -- may need a larger LR.
# 14. Stride: step size of the kernel; padding: border pixels added,
#     typically to preserve output size. O = floor((N+2P-K)/S) + 1.
# 15. MobileNetV2 is efficient due to depthwise separable convolutions and
#     inverted residual blocks with linear bottlenecks.
# 16. Depthwise separable convolution = depthwise (per-channel) conv +
#     pointwise (1x1) conv, far fewer params than a standard conv.
# 17. Transfer learning: reusing a model/representations learned on one
#     task as the starting point for a related task.
# 18. Feature extraction freezes the whole base, only the head trains;
#     fine-tuning also unfreezes/trains some base layers.
# 19. Fine-tuning uses a smaller LR to avoid catastrophic forgetting of the
#     pretrained weights.
# 20. K-Fold CV gives a lower-variance, more reliable performance estimate
#     than one train/val split.
# 21. The test set must stay untouched during tuning or its reported
#     performance becomes optimistically biased.
# 22. Reporting mean + SD together shows both typical performance and
#     consistency across splits.
# 23. Highest validation accuracy alone isn't always sufficient -- could be
#     a lucky split, hide high variance, or cost much more compute; weigh
#     mean, SD, cost, and confirmed test performance together.


# ==========================================================================
# 16. Additional Exercise
# ==========================================================================
EXTRA_CONFIGS = {
    "E1 (aggressive fine-tune, low dropout)": dict(
        init="he_normal", dropout_rate=0.1, use_batchnorm=False, l2_reg=0.0,
        optimizer="adam", learning_rate=1e-5,
        trainable_base=True, fine_tune_at=50,
    ),
    "E2 (feature-extraction, high dropout, RMSProp)": dict(
        init="glorot_uniform", dropout_rate=0.6, use_batchnorm=True, l2_reg=0.0,
        optimizer="rmsprop", learning_rate=5e-4,
        trainable_base=False,
    ),
}
EXTRA_BATCH_SIZES = {
    "E1 (aggressive fine-tune, low dropout)": 16,
    "E2 (feature-extraction, high dropout, RMSProp)": 64,
}

extra_cv_results = load_ckpt("extra_cv_results")
for name in EXTRA_CONFIGS:
    extra_cv_results.setdefault(name, [])

for name, cfg in EXTRA_CONFIGS.items():
    bsz = EXTRA_BATCH_SIZES[name]
    print(f"=== Cross-validating {name} (batch_size={bsz}) ===")
    run_cv_folds(name, cfg, batch_size=bsz, results_dict=extra_cv_results, ckpt_name="extra_cv_results")

comparison_rows = []
for name, accs in extra_cv_results.items():
    comparison_rows.append({
        "Configuration": name,
        "Mean CV Accuracy (%)": round(np.mean(accs), 2),
        "SD": round(np.std(accs), 2),
    })
comparison_rows.append({
    "Configuration": f"Previously selected: {best_config_name}",
    "Mean CV Accuracy (%)": round(np.mean(cv_results[best_config_name]), 2),
    "SD": round(np.std(cv_results[best_config_name]), 2),
})
additional_exercise_table = pd.DataFrame(comparison_rows)
print(additional_exercise_table)

fig, ax = plt.subplots(figsize=(7, 5))
names = additional_exercise_table["Configuration"]
means_ = additional_exercise_table["Mean CV Accuracy (%)"]
sds_ = additional_exercise_table["SD"]
ax.bar(names, means_, yerr=sds_, capsize=6, color=["tab:orange", "tab:green", "tab:blue"], alpha=0.85)
ax.set_ylabel("Mean Validation Accuracy (%)")
ax.set_title("Additional Exercise: New Configurations vs. Previously Selected")
plt.xticks(rotation=20, ha="right")
ax.grid(alpha=0.3, axis="y")
save_fig(fig, "plot_additional_exercise_comparison")

# Justification: compare accuracy, SD, compute cost, and (if a new config
# wins) re-confirm on the test set before adopting it over best_config_name.


# ==========================================================================
# 17. Expected Outcome
# ==========================================================================
# This script demonstrates, section by section, how weight initialization,
# regularization, optimizer choice, and hyperparameters shape MobileNetV2's
# performance on 37-way pet breed classification, then shows fine-tuning
# beats plain feature extraction, and finally selects/justifies one final
# configuration via 5-fold cross-validation (mean + SD), confirmed once on
# an untouched test set, with per-class performance examined via the
# confusion matrix. Section 16 repeats the selection process on two new
# candidate configurations to show it generalises.

print("\nAll sections complete. Figures are in ./figures, checkpoints in ./checkpoints.")
