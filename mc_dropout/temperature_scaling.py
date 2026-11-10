"""
Step 3: does temperature scaling (Guo et al., 2017) fix the same problem as MC dropout?

Guo et al. found that modern neural networks are overconfident, and that dividing the logits
by a single learned temperature T before the softmax fixes most of the miscalibration.

This script compares three versions of the same classifier:
  - Baseline: the notebook's model, no dropout
  - Temperature scaling: the baseline with T fitted on held-out data (Guo et al.)
  - MC dropout: Dropout(0.5) kept on at prediction time, 30 passes averaged

and measures:
  - Expected Calibration Error (ECE) on dogs, the metric Guo et al. use
  - mean confidence on 100 photos of people
  - AUROC for telling dogs from people using the model's confidence
    (0.5 = can't tell them apart, 1.0 = perfect separation)

The 20% validation split is cut in half: one half for early stopping and fitting T,
the other half (never seen during training or fitting) for every number reported.

Usage: python temperature_scaling.py [seed]
"""
import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
DATA_DIR = "../dog-breed-identification/"
PASSES = 30

labels = pd.read_csv(DATA_DIR + "labels.csv")["breed"]
breeds = np.unique(labels)
y = np.array([label == breeds for label in labels]).astype("float32")
X = np.concatenate([np.load(f) for f in sorted(glob.glob("features_dogs_*.npy"))])
people = np.load("features_people.npy")

# same 80/20 split as mc_dropout.py, then the 20% is halved into calibration and test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_cal, X_test, y_cal, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)


def train(dropout_rate):
    tf.keras.utils.set_random_seed(SEED)
    layers = [tf.keras.Input(shape=(X.shape[1],))]
    if dropout_rate:
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(len(breeds)))  # logits; softmax applied below
    model = tf.keras.Sequential(layers)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_cal, y_cal), epochs=100, batch_size=32, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)])
    return model


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, onehot, bins=15):
    """Expected Calibration Error with 15 equal-width bins, as in Guo et al."""
    conf, correct = probs.max(1), probs.argmax(1) == onehot.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def report(name, dog_probs, people_probs):
    acc = (dog_probs.argmax(1) == y_test.argmax(1)).mean()
    scores = np.concatenate([dog_probs.max(1), people_probs.max(1)])
    is_dog = np.concatenate([np.ones(len(dog_probs)), np.zeros(len(people_probs))])
    row = {"method": name, "dog accuracy": acc, "ECE (dogs)": ece(dog_probs, y_test),
           "confidence on dogs": dog_probs.max(1).mean(),
           "confidence on people": people_probs.max(1).mean(),
           "AUROC dogs vs people": roc_auc_score(is_dog, scores)}
    print(f"{name:20s} acc {row['dog accuracy']:.1%} | ECE {row['ECE (dogs)']:.3f} | "
          f"conf dogs {row['confidence on dogs']:.1%} | conf people {row['confidence on people']:.1%} | "
          f"AUROC {row['AUROC dogs vs people']:.3f}")
    return row


baseline = train(0)
logits_cal, logits_test, logits_people = (baseline.predict(x, verbose=0) for x in (X_cal, X_test, people))

# fit the temperature by minimising negative log likelihood on the calibration half
def nll(T):
    p = softmax(logits_cal / T)
    return -np.mean(np.log(p[np.arange(len(p)), y_cal.argmax(1)] + 1e-12))

T = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
print(f"Fitted temperature T = {T:.2f}")

rows = [report("Baseline", softmax(logits_test), softmax(logits_people)),
        report("Temperature scaling", softmax(logits_test / T), softmax(logits_people / T))]

mc = train(0.5)
mc_probs = lambda x: np.mean([softmax(mc(x, training=True).numpy()) for _ in range(PASSES)], axis=0)
rows.append(report("MC dropout", mc_probs(X_test), mc_probs(people)))

pd.DataFrame(rows).assign(seed=SEED, T=T).to_csv(f"temperature_results_seed{SEED}.csv", index=False)
