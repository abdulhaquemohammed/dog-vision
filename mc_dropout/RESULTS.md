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
pip install tensorflow tf-keras tensorflow-hub pandas scikit-learn
python extract_features.py people
python extract_features.py dogs 0 2500
python extract_features.py dogs 2500 5000
python extract_features.py dogs 5000 7500
python extract_features.py dogs 7500 10222
python mc_dropout.py 42      # repeat with seeds 1, 2, 3
```

Feature extraction runs the frozen hub layer once and caches its outputs, so each training run then takes a few seconds on a CPU. Results here were produced on CPU with TensorFlow 2.21.
