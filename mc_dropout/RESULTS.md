# MC dropout experiment

**Question:** when the classifier sees a photo of a person, it still confidently names a dog breed. Can MC dropout make it show that it's unsure?

## Setup

- **Model:** the same as the notebook: TF Hub `mobilenet_v2_130_224` (frozen) followed by a 120-unit softmax layer.
- **Data:** all 10,222 labelled Kaggle images, split 80/20 into training and validation (the notebook's full model had no validation split).
- **Baseline:** no dropout.
- **MC dropout:** `Dropout(0.5)` before the output layer, left on at prediction time, with each image's softmax output averaged over 30 passes.
- **Out-of-distribution test set:** 100 photos of 100 different people from [Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/), picked at random with a fixed seed (`people_used.txt`).
- **Repeats:** 4 training runs with different seeds, because results vary a lot from run to run.

## Results

| Seed | Dog accuracy (baseline) | Dog accuracy (MC) | Confidence on people (baseline) | Confidence on people (MC) | People under 50% confidence (baseline) | People under 50% confidence (MC) |
|---|---|---|---|---|---|---|
| 42 | 81.4% | 80.8% | 64.6% | 30.4% | 33% | 91% |
| 1 | 80.5% | 80.8% | 61.2% | 36.7% | 37% | 79% |
| 2 | 80.9% | 80.6% | 48.4% | 37.6% | 56% | 83% |
| 3 | 81.3% | 81.2% | 54.5% | 25.0% | 48% | 96% |
| **Mean** | **81.0%** | **80.9%** | **57.2%** | **32.4%** | **44%** | **87%** |

"Confidence" is the mean of the highest softmax probability.

## What it shows

- **MC dropout roughly halves the confidence on people (57% to 32%) without hurting dog accuracy (81% either way).** Over 4 runs, the share of people photos the model was less than 50% sure about went from 44% to 87%.
- **It also lowers confidence on dogs (about 88% to 81%).** Part of that is the model being honest about breeds it confuses, but it means a single confidence threshold can't perfectly separate dogs from not-dogs.
- **The baseline varies a lot between runs (48% to 65% on people).** A single run would have overstated or understated the effect, which is why the headline numbers are averaged.

## Limitations

- Dropout is only on the new output layer, not inside MobileNetV2, so it captures uncertainty in the last layer only.
- Early stopping used the same validation set the accuracy is reported on, so dog accuracy is slightly optimistic for both models equally.
- LFW photos are tightly cropped faces. Other kinds of non-dog images (cats, objects) might behave differently.

## Running it

From this folder, with the Kaggle data in `../dog-breed-identification/`:

```
pip install tensorflow tf-keras tensorflow-hub pandas scikit-learn scipy
python extract_features.py people
python extract_features.py dogs 0 2500
python extract_features.py dogs 2500 5000
python extract_features.py dogs 5000 7500
python extract_features.py dogs 7500 10222
python mc_dropout.py 42      # repeat with seeds 1, 2, 3
python temperature_scaling.py 42   # repeat with seeds 1, 2, 3 (also needs scipy)
```

Feature extraction runs the frozen hub layer once and caches its outputs, so each training run then takes a few seconds on a CPU. Results here were produced on CPU with TensorFlow 2.21.

---

# Follow-up: temperature scaling vs MC dropout

**Question:** [Guo et al. (2017), "On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599) found that modern networks are overconfident and that a single parameter, a temperature T dividing the logits before the softmax, fixes most of it. Does that also stop the model being confident about photos of people?

## Setup

`temperature_scaling.py`. Same features and 80/20 split as above, but the 20% validation set is cut in half:
- **Calibration half (1,022 images):** early stopping, and fitting T by minimising negative log likelihood, as in the paper.
- **Test half (1,022 images):** every number below. Nothing here was used for training or fitting.

Metrics:
- **ECE** (Expected Calibration Error, 15 bins), the paper's metric: the average gap between how confident the model is and how often it's right. Lower is better.
- **AUROC dogs vs people:** how well the model's confidence alone separates dog photos from people photos. 0.5 means it can't tell them apart; 1.0 means perfect separation.

## Results (mean of 4 seeds: 42, 1, 2, 3)

| Method | Dog accuracy | ECE (dogs) | Confidence on dogs | Confidence on people | AUROC dogs vs people |
|---|---|---|---|---|---|
| Baseline | 81.8% | 0.061 | 87.5% | 53.7% | 0.881 |
| Temperature scaling | 81.8% | 0.029 | 81.9% | 40.2% | 0.907 |
| MC dropout | 81.7% | 0.029 | 80.7% | 32.8% | 0.944 |

Fitted temperatures were 1.19 to 1.53 (T > 1 means the baseline was overconfident, as the paper predicts).

## What it shows

- **Temperature scaling reproduces the paper's result.** Calibration error on dogs halved (0.061 to 0.029) with accuracy unchanged, because dividing by T never changes which breed is ranked first.
- **But it mostly lowers confidence on everything.** Dogs and people both drop, so separation only improves from 0.88 to 0.91.
- **MC dropout matches its calibration and separates better.** Same ECE (0.029), similar confidence on dogs (81%), but people fall further (33% vs 40%) and AUROC reaches 0.94. Averaging over many dropout masks spreads the probability out more on images that don't look like the training data.
- **Why:** temperature scaling is one number applied to every image, so it fixes overconfidence on average. It has no way to treat an unfamiliar image differently from a familiar one. MC dropout's uncertainty comes from disagreement between passes, which is larger on images unlike the training data.

## Limitations

- Same as above: dropout only on the output layer, LFW faces only, a 1,022-image test set.
- The baseline numbers here differ slightly from the first table because early stopping now uses a different held-out set.
