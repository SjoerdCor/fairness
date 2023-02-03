import sys

import pytest
import numpy as np

from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import clone as skclone
from sklearn.ensemble import RandomForestClassifier

sys.path.append("..")
from fairness import fairestimator
import importlib

importlib.reload(fairestimator)

# Test default parameters
clf = RandomForestClassifier(min_samples_leaf=10, max_depth=3, random_state=42)


def test_all_columns_ignored():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    ib = fairestimator.IgnoringBiasClassifier(skclone(clf), ignored_cols=[0])

    ib.fit(X, y)
    result = ib._prepare_new_dataset(X)
    expected = np.array([[2.5], [2.5], [2.5], [2.5]])
    assert np.array_equal(result, expected)
