import sys

import pytest
import numpy as np
import pandas as pd

from sklearn.base import clone as skclone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris

sys.path.append(r"..\..")
sys.path.append("..")
from fairness import fairestimator
import importlib

importlib.reload(fairestimator)

clf = RandomForestClassifier(min_samples_leaf=10, max_depth=3, random_state=42)
regressor = RandomForestRegressor(min_samples_leaf=10, max_depth=3, random_state=42)

# TODO: write docstrings for each test
def data(as_dataframe=False):
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    y = np.array([1] * 4)
    if as_dataframe:
        X = pd.DataFrame(X)
        y = pd.Series(y)
    return X, y


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_determine_no_cols_to_correct(as_dataframe):
    X, y = data(as_dataframe)
    ib = fairestimator.IgnoringBiasClassifier(skclone(clf), ignored_cols=None)
    ib.fit(X, y)
    assert ib.ignored_cols_ == []


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_determine_cols_to_correct(as_dataframe):
    X, y = data(as_dataframe)
    ib = fairestimator.IgnoringBiasClassifier(skclone(clf), ignored_cols=[0, 1])
    ib.fit(X, y)
    assert ib.ignored_cols_ == [0, 1]


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_impute_fixed_value(as_dataframe):
    X, y = data(as_dataframe)
    ib = fairestimator.IgnoringBiasClassifier(
        skclone(clf), ignored_cols=[0], impute_values=[2.5]
    )
    ib.fit(X, y)

    result = ib._prepare_new_dataset(X)
    expected = np.array([[2.5, 2], [2.5, 4], [2.5, 6], [2.5, 8]])
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_impute_fixed_value_multiple(as_dataframe):
    X, y = data(as_dataframe)
    ib = fairestimator.IgnoringBiasClassifier(
        skclone(clf), ignored_cols=[0, 1], impute_values=[2.5, 5]
    )
    ib.fit(X, y)

    result = ib._prepare_new_dataset(X)
    expected = np.array([[2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5]])
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_calculate_correct_mean_for_imputation(as_dataframe):
    X, y = data(as_dataframe)
    ib = fairestimator.IgnoringBiasClassifier(
        skclone(clf), ignored_cols=[0], impute_values=None
    )

    ib.fit(X, y)
    result = ib.impute_values_
    expected = np.array([2.5], dtype=np.float64)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_calculate_correct_mean_for_imputation_multiple(as_dataframe):
    X, y = data(as_dataframe)
    ib = fairestimator.IgnoringBiasClassifier(
        skclone(clf), ignored_cols=[0, 1], impute_values=None
    )

    ib.fit(X, y)
    result = ib.impute_values_
    expected = np.array([2.5, 5], dtype=np.float64)
    assert np.array_equal(result, expected)


def test_uncorrect_predictions_regressor():
    X, y = load_iris(return_X_y=True)
    ir = fairestimator.IgnoringBiasRegressor(skclone(regressor), ignored_cols=None)
    ir.fit(X, y)
    regres = skclone(regressor).fit(X, y)
    assert np.array_equal(ir._calculate_uncorrected_predictions(X), regres.predict(X))
