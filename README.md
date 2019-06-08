# Confusion Matrix

A simple pandas/numpy-based module for constructing, analysing, and plotting confusion matrices that I found myself copying between projects so often that I decided to make it a separate, pip-installable package. It was written with **python 3.6**, but \_\_future\_\_ statements have been included and f-string have been avoided intentionally. 
## Installation 

Install via pip (from PyPi)
```
pip install cmat
```
Or, alternatively, just copy the [main module](cmat/confusion_matrix.py) into your project and hardcode all config.xx values.

## Usage

The functionality should be intuitive for anyone familiar with the confusion_matrix and classification_report from [scikit-learn](https://scikit-learn.org/stable/). Below follow basic instructions for instantiating the class.

To create a confusion matrix, do:
```python
import cmat

cm = cmat.ConfusionMatrix.create(
  # 1D arrray with ground truth labels
  y_true = ground_truth,
  # 2D array with predictions
  y_pred = predictions
  # (optional) List of values that might occur in y_true/y_pred
  labels = class_labels
  # (optional) List of names corresponding to labels
  names = class_names
)
```
From there, you can get several metrics. A supplementary [jupyter notebook](notebooks/demo.ipynb) has been included for more in depth documentation.


## Development 

Since the project is so small, no CI features have been added. There is, however, a small test suite which at the time of writing got 66% coverage on the main script.j Pylint has not been applied because it hates my intentation style.

Feel free to fork, post issues, or make pull requests if you have any suggestions to improvements.

Some possible TODOs include:

- Plot for comparing multiple confusion matrices w.r.t. a specific metric. This could be useful for e.g. cross-validation comparisons
- Support for adding sample or class weight when creating the matrix
