"""
Train a U-Net for cattle detection (binary segmentation) on SkyFi imagery.

Loads patches.npy / masks.npy created by prepare_data.py, builds a U-Net
encoder-decoder, trains with binary focal loss, and saves the best model.

Usage:
    train.bat                        # defaults: 30 epochs, batch 32
    train.bat --epochs 2             # smoke test
    train.bat --epochs 50 --batch 16
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings
warnings.filterwarnings("ignore")

import h5py
import psutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import run_logging

HERE = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR = os.environ.get("TF_COWDETECT_ROOT", HERE)
DATA_DIR = os.path.join(EXTERNAL_DIR, "data")
MODELS_DIR = os.path.join(EXTERNAL_DIR, "models")

VAL_SPLIT = 0.2
SEED = 1337


# ---------------------------------------------------------------------------
#  CleanProgress callback
# ---------------------------------------------------------------------------
class CleanProgress(keras.callbacks.Callback):
    """Plain-text progress for cmd.exe."""

    def __init__(self, total_epochs: int, update_every_sec: float = 5.0):
        super().__init__()
        self.total_epochs = total_epochs
        self.update_every_sec = update_every_sec
        self._run_start = 0.0
        self._epoch_start = 0.0
        self._last_print = 0.0
        self._epoch_durations: list[float] = []

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        h, r = divmod(seconds, 3600)
        m, s = divmod(r, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    @staticmethod
    def _get(logs: dict, key: str) -> str:
        v = (logs or {}).get(key)
        if v is None:
            return "  n/a "
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return "  n/a "

    def on_train_begin(self, logs=None):
        self._run_start = time.time()
        print(f"[run] starting {self.total_epochs} epochs", flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()
        self._last_print = 0.0
        overall_pct = (epoch * 100) // self.total_epochs
        elapsed = time.time() - self._run_start
        if self._epoch_durations:
            avg = sum(self._epoch_durations) / len(self._epoch_durations)
            eta = (self.total_epochs - epoch) * avg
            eta_str = f", ETA {self._fmt_time(eta)}"
        else:
            eta_str = ""
        print(
            f"\n[epoch {epoch + 1:2d}/{self.total_epochs}] start  "
            f"(overall {overall_pct:3d}%, elapsed {self._fmt_time(elapsed)}{eta_str})",
            flush=True,
        )

    def on_train_batch_end(self, batch, logs=None):
        now = time.time()
        if now - self._last_print < self.update_every_sec:
            return
        self._last_print = now
        steps = (self.params or {}).get("steps") or 0
        pct = ((batch + 1) * 100 // steps) if steps else 0
        epoch_elapsed = now - self._epoch_start
        print(
            f"  batch {batch + 1:4d}/{steps:4d} ({pct:3d}%)  "
            f"loss={self._get(logs, 'loss')}  recall={self._get(logs, 'recall')}  "
            f"[{self._fmt_time(epoch_elapsed)}]",
            flush=True,
        )

    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        duration = now - self._epoch_start
        self._epoch_durations.append(duration)
        print(
            f"[epoch {epoch + 1:2d}/{self.total_epochs}] done in {self._fmt_time(duration)}  "
            f"loss={self._get(logs, 'loss')}  "
            f"val_loss={self._get(logs, 'val_loss')}  "
            f"val_f1={self._get(logs, 'val_f1')}  "
            f"val_recall={self._get(logs, 'val_recall')}  "
            f"val_precision={self._get(logs, 'val_precision')}",
            flush=True,
        )

    def on_train_end(self, logs=None):
        total = time.time() - self._run_start
        print(f"\n[run] complete in {self._fmt_time(total)}", flush=True)


# ---------------------------------------------------------------------------
#  Backup
# ---------------------------------------------------------------------------
def backup_existing_model(model_path: str) -> None:
    if not os.path.exists(model_path):
        return
    backup_dir = os.path.join(MODELS_DIR, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(os.path.basename(model_path))
    shutil.copy2(model_path, os.path.join(backup_dir, f"{base}_{timestamp}{ext}"))
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(backup_dir, f"metadata_{timestamp}.json"))
    print(f"[backup] previous model backed up to {backup_dir}")


# ---------------------------------------------------------------------------
#  Binary focal loss
# ---------------------------------------------------------------------------
def binary_focal_loss(alpha: float = 0.75, gamma: float = 2.0):
    """Focal loss for binary segmentation — down-weights easy background pixels."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # alpha weighting: alpha for positive, (1-alpha) for negative
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        # focal weighting
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(alpha_t * focal_weight * bce)
    return loss_fn


def tversky_loss(alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
    """Soft-Tversky loss. beta > alpha penalizes false negatives harder than
    false positives, directly pushing recall up — the fix for our
    high-precision / low-recall regime. alpha=beta=0.5 reduces to Dice."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1.0 - tversky
    return loss_fn


def combined_loss(focal_alpha: float = 0.75, focal_gamma: float = 2.0,
                  tversky_alpha: float = 0.3, tversky_beta: float = 0.7,
                  tversky_weight: float = 1.0):
    """Pixel-level focal term + region-level Tversky term. Focal keeps the
    per-pixel hard-example signal; Tversky (beta>alpha) supplies the
    recall-oriented region penalty."""
    focal = binary_focal_loss(focal_alpha, focal_gamma)
    tversky = tversky_loss(tversky_alpha, tversky_beta)
    def loss_fn(y_true, y_pred):
        return focal(y_true, y_pred) + tversky_weight * tversky(y_true, y_pred)
    return loss_fn


class F1Score(keras.metrics.Metric):
    """Streaming per-pixel F1 at a fixed threshold — used for checkpoint /
    early-stopping selection so we optimize the precision-recall balance
    rather than recall alone (which is trivially maxed by over-predicting)."""

    def __init__(self, threshold: float = 0.5, name: str = "f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1.0 - y_pred)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        return 2.0 * precision * recall / (precision + recall + 1e-7)

    def reset_state(self):
        for v in self.variables:
            v.assign(0.0)


# ---------------------------------------------------------------------------
#  U-Net model
# ---------------------------------------------------------------------------
def conv_block(x, filters: int):
    """Two Conv3x3 + BN + ReLU layers."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_unet(patch_size: int, num_bands: int, tversky_beta: float = 0.7) -> keras.Model:
    inputs = keras.Input(shape=(patch_size, patch_size, num_bands), name="patch")

    # NOTE: augmentation is applied jointly to (image, mask) in the tf.data
    # pipeline (see augment_pair). It must NOT live inside the model graph:
    # flipping/rotating only the input there desynchronizes it from the label
    # and teaches an orientation-averaged mapping that destroys recall.

    # Encoder
    enc1 = conv_block(inputs, 32)  # 64x64
    p1 = layers.MaxPooling2D(2)(enc1)  # 32x32

    enc2 = conv_block(p1, 64)     # 32x32
    p2 = layers.MaxPooling2D(2)(enc2)  # 16x16

    enc3 = conv_block(p2, 128)    # 16x16
    p3 = layers.MaxPooling2D(2)(enc3)  # 8x8

    enc4 = conv_block(p3, 128)    # 8x8
    p4 = layers.MaxPooling2D(2)(enc4)  # 4x4

    # Bottleneck
    bn = conv_block(p4, 256)      # 4x4

    # Decoder
    up4 = layers.UpSampling2D(2)(bn)           # 8x8
    up4 = layers.Concatenate()([up4, enc4])
    dec4 = conv_block(up4, 128)

    up3 = layers.UpSampling2D(2)(dec4)         # 16x16
    up3 = layers.Concatenate()([up3, enc3])
    dec3 = conv_block(up3, 128)

    up2 = layers.UpSampling2D(2)(dec3)         # 32x32
    up2 = layers.Concatenate()([up2, enc2])
    dec2 = conv_block(up2, 64)

    up1 = layers.UpSampling2D(2)(dec2)         # 64x64
    up1 = layers.Concatenate()([up1, enc1])
    dec1 = conv_block(up1, 32)

    # Output: per-pixel probability
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="cow_prob")(dec1)

    model = keras.Model(inputs, outputs, name="cowdetect_unet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss(focal_alpha=0.75, focal_gamma=2.0,
                           tversky_alpha=1.0 - tversky_beta, tversky_beta=tversky_beta),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            F1Score(name="f1"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
#  Memory helpers
# ---------------------------------------------------------------------------
def print_memory(label: str = "") -> None:
    proc = psutil.Process()
    rss = proc.memory_info().rss / 1e9
    vm = psutil.virtual_memory()
    print(f"[mem] {label:20s}  process={rss:.2f} GB  system={vm.percent:.0f}% "
          f"({vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB)", flush=True)


def check_memory_budget(max_pct: float) -> bool:
    return psutil.virtual_memory().percent < max_pct


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--tversky-beta", type=float, default=0.7,
                    help="Tversky FN penalty (alpha=1-beta). Higher = push recall harder. "
                         "0.7 balanced, 0.8 recall-leaning.")
    p.add_argument("--isolated-boost", type=int, default=1,
                    help="Oversample factor for patches with a single (isolated) cow. "
                         "The model misses lone cows; >1 up-weights them. 1=off, 3 recommended.")
    p.add_argument("--spectral-bands", type=int, default=4,
                    help="Number of leading raw spectral bands that receive photometric "
                         "(brightness/contrast) jitter. Derived channels beyond this (e.g. NDVI) "
                         "get geometric augmentation only — jittering a ratio index is meaningless.")
    p.add_argument("--model-name", default="cowdetect_unet.keras",
                    help="Model filename (saved under models/)")
    p.add_argument("--data-dir", default=DATA_DIR, help="Directory with patches.npy / masks.npy")
    p.add_argument("--max-memory-pct", type=float, default=80.0,
                    help="Abort if system memory exceeds this %%")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_logging.setup_tee("train")
    print("=" * 60)
    print("  TF_CowDetection  |  TRAINING (U-Net)")
    print("=" * 60)
    os.makedirs(MODELS_DIR, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logdir = os.path.join(EXTERNAL_DIR, "runs", run_ts)

    print(f"TensorFlow : {tf.__version__}   Keras : {keras.__version__}")
    print(f"GPUs       : {tf.config.list_physical_devices('GPU') or 'none (CPU)'}")
    print(f"Memory cap : {args.max_memory_pct:.0f}%")
    print_memory("startup")

    # --- Load data ---
    patches_path = os.path.join(args.data_dir, "patches.npy")
    masks_path = os.path.join(args.data_dir, "masks.npy")
    if not os.path.exists(patches_path) or not os.path.exists(masks_path):
        print(f"ERROR: patches.npy / masks.npy not found in {args.data_dir}")
        print("       Run prepare_data.bat first.")
        return 1

    patches = np.load(patches_path, mmap_mode="r")  # (N, 64, 64, 4)
    masks = np.load(masks_path, mmap_mode="r")       # (N, 64, 64, 1)
    n, patch_size, _, num_bands = patches.shape
    print(f"Loaded     : {n:,} patches, shape={patches.shape}, dtype={patches.dtype}")
    print_memory("after load (mmap)")

    # --- Stratified split: by whether patch contains a cow ---
    has_cow = np.array([np.any(masks[i] > 0) for i in range(n)])
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    idx_train, idx_val = train_test_split(
        indices, test_size=VAL_SPLIT, random_state=SEED, stratify=has_cow
    )
    print(f"Train      : {len(idx_train):,}   Val: {len(idx_val):,}")

    # Copy splits as float32
    X_train = patches[idx_train].astype(np.float32)
    Y_train = masks[idx_train].astype(np.float32)
    print_memory("after X_train f32")

    if not check_memory_budget(args.max_memory_pct):
        print(f"ERROR: System memory at {psutil.virtual_memory().percent:.0f}%")
        return 1

    X_val = patches[idx_val].astype(np.float32)
    Y_val = masks[idx_val].astype(np.float32)
    del patches, masks, indices
    print_memory("after X_val f32")

    if not check_memory_budget(args.max_memory_pct):
        print(f"ERROR: System memory at {psutil.virtual_memory().percent:.0f}%")
        return 1

    # --- Per-band normalization ---
    band_mean = np.mean(X_train, axis=(0, 1, 2))  # (4,)
    band_std = np.std(X_train, axis=(0, 1, 2))     # (4,)
    band_std[band_std == 0] = 1.0
    print(f"Band means : {band_mean}")
    print(f"Band stds  : {band_std}")

    X_train = (X_train - band_mean) / band_std
    X_val = (X_val - band_mean) / band_std

    # --- Isolation-aware oversampling (training set only) -------------------
    # The miss analysis showed the model fails on ISOLATED cows (lone animals
    # are ~3x rarer among detections than herd cows). We count distinct cow
    # blobs per patch as a local-density proxy and repeat sparse-cow patches so
    # lone/small-group cows are seen more often. Val is left untouched (honest).
    if args.isolated_boost > 1:
        from scipy import ndimage

        def patch_factor(mask_2d):
            n_blobs = ndimage.label(mask_2d > 0)[1]
            if n_blobs == 0:
                return 1                       # background
            if n_blobs == 1:
                return args.isolated_boost     # isolated cow
            if n_blobs <= 3:
                return 2                       # small group
            return 1                           # herd

        reps = np.array([patch_factor(Y_train[i, :, :, 0]) for i in range(len(Y_train))],
                        dtype=np.int64)
        n_iso = int(np.sum(reps == args.isolated_boost))
        n_grp = int(np.sum(reps == 2))
        rep_idx = np.repeat(np.arange(len(Y_train)), reps)
        np.random.default_rng(SEED).shuffle(rep_idx)
        X_train = X_train[rep_idx]
        Y_train = Y_train[rep_idx]
        print(f"Oversampling : isolated={n_iso} (x{args.isolated_boost}), "
              f"small-group={n_grp} (x2)  ->  train {len(rep_idx):,} patches")
        print_memory("after oversample")

    # --- Build tf.data pipelines ---
    AUTOTUNE = tf.data.AUTOTUNE

    n_spectral = min(args.spectral_bands, num_bands)

    def augment_pair(x, y):
        """Geometric augmentation (flips + 90-degree rotations) applied jointly
        and losslessly to image AND mask. Photometric jitter (brightness/
        contrast) is applied to the image ONLY, and only to the leading
        n_spectral raw bands — derived channels like NDVI are ratio indices
        that brightness/contrast would corrupt, so they get geometry only.
        Data is per-band standardized, so brightness is an additive shift
        (~sigma units) and contrast a multiplicative gain about the band mean."""
        # --- geometric: joint (image + mask), all channels ---
        if tf.random.uniform(()) < 0.5:
            x = tf.image.flip_left_right(x)
            y = tf.image.flip_left_right(y)
        if tf.random.uniform(()) < 0.5:
            x = tf.image.flip_up_down(x)
            y = tf.image.flip_up_down(y)
        k = tf.random.uniform((), maxval=4, dtype=tf.int32)
        x = tf.image.rot90(x, k)
        y = tf.image.rot90(y, k)
        # --- photometric: image only, spectral channels only ---
        if n_spectral < num_bands:
            spec = x[..., :n_spectral]
            extra = x[..., n_spectral:]
            spec = tf.image.random_brightness(spec, max_delta=0.2)
            spec = tf.image.random_contrast(spec, lower=0.8, upper=1.2)
            x = tf.concat([spec, extra], axis=-1)
        else:
            x = tf.image.random_brightness(x, max_delta=0.2)
            x = tf.image.random_contrast(x, lower=0.8, upper=1.2)
        return x, y

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        .shuffle(min(10000, len(X_train)), seed=SEED)
        .map(augment_pair, num_parallel_calls=AUTOTUNE)
        .batch(args.batch)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        .batch(args.batch)
        .prefetch(AUTOTUNE)
    )
    del X_train, X_val, Y_train, Y_val
    print_memory("after tf.data")

    # --- Build model ---
    print(f"Tversky beta : {args.tversky_beta} (alpha={1.0 - args.tversky_beta:.2f})")
    model = build_unet(patch_size, num_bands, tversky_beta=args.tversky_beta)
    model.summary()

    # --- Backup ---
    model_path = os.path.join(MODELS_DIR, args.model_name)
    backup_existing_model(model_path)

    # --- Train ---
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_f1", patience=8, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_f1",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=tb_logdir, update_freq="epoch"),
        CleanProgress(total_epochs=args.epochs, update_every_sec=5.0),
    ]
    print(f"[tensorboard] logging to {tb_logdir}")
    print(f"[tensorboard] view: tensorboard --logdir {os.path.join(EXTERNAL_DIR, 'runs')}  -> http://localhost:6006")

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
    )
    dt = time.time() - t0
    print(f"\nTraining took {dt:.1f}s")
    print_memory("after training")

    results = model.evaluate(val_ds, verbose=0)
    for name, val in zip(model.metrics_names, results):
        print(f"Final val {name:10s}: {val:.4f}")
    print(f"Best model saved  : {model_path}")

    # --- Save metadata ---
    meta = {
        "task": "cow_detection",
        "model_type": "unet",
        "num_bands": num_bands,
        "patch_size": patch_size,
        "tversky_beta": args.tversky_beta,
        "isolated_boost": args.isolated_boost,
        "band_mean": band_mean.tolist(),
        "band_std": band_std.tolist(),
        "train_size": len(idx_train),
        "val_size": len(idx_val),
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved    : {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
