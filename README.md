# Dog Vision 🐶

An end-to-end multi-class image classifier that predicts a dog's breed from a photo, built with TensorFlow and TensorFlow Hub.

Built while working through Zero To Mastery's Complete AI, Machine Learning and Data Science course, with my own fixes for an updated TensorFlow version and my own model experiment.

## Problem

Given a photo of a dog, identify which of **120 breeds** it is.

## Data

[Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/data):
- ~10,000 labelled training images
- ~10,000 unlabelled test images
- 120 breeds (classes)

The dataset isn't included in this repo (it's ~700 MB). To run the notebook, download it from Kaggle, upload the zip to Google Drive and update the paths in the first few cells of the notebook.

## Approach

1. **Preprocess** images into 224x224 tensors with pixel values scaled to 0-1, and turn labels into one-hot boolean arrays.
2. **Batch** the data (32 images per batch) using `tf.data`.
3. **Transfer learning:** use a pretrained MobileNetV2 model from TensorFlow Hub (trained on ImageNet) as the base, with a new 120-unit softmax output layer.
4. **Train** with TensorBoard logging and early stopping.
5. **Experiment** on a 1,000-image subset first, then train on the full dataset.
6. **Predict** on the Kaggle test set and on my own custom photos.

## Results

| Experiment | Training accuracy | Validation accuracy |
|---|---|---|
| MobileNetV2, 1,000 images | 100% | 66% |
| EfficientNetV2-B0, 1,000 images | 100% | 63% |
| MobileNetV2, full dataset (~10k images) | 99.9% | n/a (no validation split) |

### What I learned from the results

- **The model overfits heavily on small data.** On the 1,000-image subset, training accuracy hits ~100% within a few epochs while validation accuracy plateaus around 63-66%. That's only ~8 images per breed, which isn't enough to generalise.
- **Swapping the model didn't help.** Replacing MobileNetV2 with EfficientNetV2-B0 made no meaningful difference (63% vs 66%, a gap of only 6 images on a 200-image validation set). The bottleneck is the amount of data, not the model architecture.
- **Training accuracy isn't real accuracy.** The full model's 99.9% is on images it trained on. On my own photos it got some breeds right but confused similar-looking ones (e.g. a chocolate Labrador predicted as a German short-haired pointer), which fits the validation numbers far better.
- **Classifiers always pick a class.** When given a photo of a person, the model still confidently predicted a dog breed, because it has no "not a dog" option.

## Problems I ran into

- **Keras 3 vs TensorFlow Hub:** newer TensorFlow ships with Keras 3, which doesn't accept `hub.KerasLayer`. Fixed by setting `TF_USE_LEGACY_KERAS=1` before importing TensorFlow.
- **Unzipping into Google Drive was unreliable:** extracting ~20,000 images into a mounted Drive folder didn't fully sync. Fixed by keeping the zip in Drive and unzipping to Colab's local disk each session, which also made training much faster.

## Next steps

- **Data augmentation** (random flips, crops, brightness) to reduce overfitting.
- **Fine-tuning** the pretrained base model instead of only training the output layer.
- **A proper validation split on the full dataset** to get an honest accuracy figure.
- **Submit to Kaggle** to get a test-set score.

## Tools

Python, TensorFlow, TensorFlow Hub, NumPy, pandas, Matplotlib, scikit-learn, Google Colab (T4 GPU)
