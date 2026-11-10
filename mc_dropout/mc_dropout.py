"""
Step 2: does MC dropout stop the classifier being confidently wrong on photos of people?

Baseline: the notebook's model (MobileNetV2 hub layer -> 120-unit softmax), no dropout.
MC dropout: Dropout(0.5) before the output layer, kept ON at prediction time, averaging 30 passes.
Dropout used this way approximates Bayesian inference over the weights (Gal & Ghahramani, 2016),
so disagreement between passes shows up as lower averaged confidence.

Usage: python mc_dropout.py [seed]
"""
import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

# proper validation split on the full dataset (random_state matches the notebook)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def train(dropout_rate):
    tf.keras.utils.set_random_seed(SEED)
    layers = [tf.keras.Input(shape=(X.shape[1],))]
    if dropout_rate:
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(len(breeds), activation="softmax"))
    model = tf.keras.Sequential(layers)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)])
    return model


def mc_predict(model, x):
    """Average the softmax output over many passes with dropout left on."""
    return np.mean([model(x, training=True).numpy() for _ in range(PASSES)], axis=0)


def report(name, val_preds, people_preds):
    acc = (val_preds.argmax(1) == y_val.argmax(1)).mean()
    print(f"{name}: dog accuracy {acc:.1%} | mean confidence on dogs {val_preds.max(1).mean():.1%} "
          f"| on people {people_preds.max(1).mean():.1%} | people under 50%: {(people_preds.max(1) < 0.5).mean():.0%}")


baseline = train(0)
report("Baseline   ", baseline.predict(X_val, verbose=0), baseline.predict(people, verbose=0))

mc_model = train(0.5)
report("MC dropout ", mc_predict(mc_model, X_val), mc_predict(mc_model, people))
