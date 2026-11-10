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
- **Classifiers always pick a class.** When given a photo of a person, the model still confidently predicted a dog breed, because it has no "not a dog" option. I followed this up with MC dropout (below).

## Follow-up: making the model show when it's unsure

Softmax forces the 120 outputs to sum to 1, so the model has to spread all its confidence across breeds even when the photo isn't a dog. In [`mc_dropout/`](mc_dropout/) I added dropout before the output layer, kept it on at prediction time and averaged 30 passes per image (MC dropout, an approximation to Bayesian inference over the weights). Tested on 100 photos of people, averaged over 4 training runs:

| | Baseline | MC dropout |
|---|---|---|
| Dog accuracy (validation) | 81.0% | 80.9% |
| Mean confidence on photos of people | 57.2% | 32.4% |
| People photos under 50% confidence | 44% | 87% |

Confidence on people roughly halved with no loss of dog accuracy. Full setup, per-run results and limitations are in [`mc_dropout/RESULTS.md`](mc_dropout/RESULTS.md).

I then compared it with temperature scaling from [Guo et al. (2017)](https://arxiv.org/abs/1706.04599), testing on a held-out half of the validation set (mean of 4 runs):

| | Calibration error (ECE) on dogs | Confidence on people | Separating dogs from people (AUROC) |
|---|---|---|---|
| Baseline | 0.061 | 53.7% | 0.881 |
| Temperature scaling | 0.029 | 40.2% | 0.907 |
| MC dropout | 0.029 | 32.8% | 0.944 |

Temperature scaling halves calibration error, matching the paper, but because it's one number applied to every image it lowers confidence on dogs and people together. MC dropout calibrates just as well and separates the two better. Details in [`mc_dropout/RESULTS.md`](mc_dropout/RESULTS.md).

## Problems I ran into

- **Keras 3 vs TensorFlow Hub:** newer TensorFlow ships with Keras 3, which doesn't accept `hub.KerasLayer`. Fixed by setting `TF_USE_LEGACY_KERAS=1` before importing TensorFlow.
- **Unzipping into Google Drive was unreliable:** extracting ~20,000 images into a mounted Drive folder didn't fully sync. Fixed by keeping the zip in Drive and unzipping to Colab's local disk each session, which also made training much faster.

## Next steps

- **Data augmentation** (random flips, crops, brightness) to reduce overfitting.
- **Fine-tuning** the pretrained base model instead of only training the output layer.
- **A proper validation split on the full dataset** to get an honest accuracy figure. (The MC dropout experiment uses an 80/20 split and gets about 81%.)
- **Submit to Kaggle** to get a test-set score.

## Tools

Python, TensorFlow, TensorFlow Hub, NumPy, pandas, Matplotlib, scikit-learn, Google Colab (T4 GPU)
