"""
Step 1: run every image through the frozen TF Hub MobileNetV2 layer once and save its outputs.

The hub layer isn't trained, so caching its outputs gives exactly the same model as the notebook
while making each training run take seconds instead of minutes on a CPU.

Usage (from this folder):
    python extract_features.py people              # 100 photos of people (LFW)
    python extract_features.py dogs 0 2500         # dog images, in chunks
    python extract_features.py dogs 2500 5000      # ... and so on up to 10,222
"""
import os, sys, glob, random
import numpy as np
import pandas as pd

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # tensorflow_hub needs Keras 2
import tensorflow as tf
import tensorflow_hub as hub

DATA_DIR = "../dog-breed-identification/"  # Kaggle data, same folder layout as the notebook
LFW_DIR = "lfw/lfw_home/lfw_funneled/"     # downloaded by sklearn.datasets.fetch_lfw_people
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"

hub_layer = hub.KerasLayer(MODEL_URL)


def process_image(path):
    """Same preprocessing as the notebook: decode, scale to 0-1, resize to 224x224."""
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, [224, 224])


def save_features(paths, out_file):
    data = tf.data.Dataset.from_tensor_slices(paths).map(process_image).batch(64)
    features = np.concatenate([hub_layer(batch).numpy() for batch in data])
    np.save(out_file, features)
    print(f"Saved {len(paths)} images to {out_file}")


if sys.argv[1] == "people":
    if not os.path.isdir(LFW_DIR):
        from sklearn.datasets import fetch_lfw_people
        try:
            fetch_lfw_people(data_home="lfw", download_if_missing=True)
        except MemoryError:
            pass  # only the download is needed, not sklearn's in-memory array
    # one photo each of 100 different people, picked at random
    by_person = {}
    for path in sorted(glob.glob(LFW_DIR + "*/*.jpg")):
        by_person.setdefault(path.split("/")[-2], []).append(path)
    random.seed(0)
    people = [by_person[name][0] for name in random.sample(sorted(by_person), 100)]
    with open("people_used.txt", "w") as f:
        f.write("\n".join(p.replace(LFW_DIR, "") for p in people))
    save_features(people, "features_people.npy")
else:
    start, end = int(sys.argv[2]), int(sys.argv[3])
    ids = pd.read_csv(DATA_DIR + "labels.csv")["id"].tolist()[start:end]
    save_features([DATA_DIR + "train/" + i + ".jpg" for i in ids], f"features_dogs_{start:05d}.npy")
