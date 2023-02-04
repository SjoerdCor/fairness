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
    """Simple sample data for many tests

    Parameters
    ----------
    as_dataframe : bool, optional
        Whether to return the data as pd.DataFrame, by default False (np.array)

    Returns
    -------
    _pd.DataFra,e | np.array
        sample data
    """
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    y = np.array([1] * 4)
    if as_dataframe:
        X = pd.DataFrame(X)
        y = pd.Series(y)
    return X, y


@pytest.mark.parametrize("as_dataframe", [False, True], ids=["array", "dataframe"])
@pytest.mark.parametrize(
    ["ignored_cols", "expected"],
    [(None, []), ([0, 1], [0, 1])],
    ids=["InputNone", "InputList"],
)
def test_determine_cols_to_correct(as_dataframe, ignored_cols, expected):
    """Test whether fit correctly detects which columns must be ignored

    Parameters
    ----------
    as_dataframe : bool, optional
        Whether to return the data as pd.DataFrame, by default False (np.array)
    """
    X, y = data(as_dataframe)
    ib = fairestimator.IgnoringBiasClassifier(skclone(clf), ignored_cols=ignored_cols)
    ib.fit(X, y)
    assert ib.ignored_cols_ == expected


@pytest.mark.parametrize("as_dataframe", [False, True], ids=["array", "dataframe"])
@pytest.mark.parametrize(
    ["impute_values", "expected"],
    [
        ([2.5], [[2.5, 2], [2.5, 4], [2.5, 6], [2.5, 8]]),
        ([2.5, 5], [[2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5]]),
    ],
    ids=["single_impute_value", "multiple_impute_values"],
)
def test_impute_fixed_value(as_dataframe, impute_values, expected):
    """Test whether imputing fixed values generates the correct dataset

    Parameters
    ----------
    as_dataframe : bool, optional
        Whether to return the data as pd.DataFrame, by default False (np.array)
    impute_values : lst
        List of values to impute
    expected : Iterable
        The generated (corrected) data
    """
    X, y = data(as_dataframe)
    ignored_cols = list(range(len(impute_values)))
    ib = fairestimator.IgnoringBiasClassifier(
        skclone(clf), ignored_cols=ignored_cols, impute_values=impute_values
    )
    ib.fit(X, y)

    result = ib._prepare_new_dataset(X)
    expected = np.array(expected)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("as_dataframe", [False, True], ids=["array", "dataframe"])
@pytest.mark.parametrize(
    ["ignored_cols", "expected"],
    [([0], [2.5]), ([0, 1], [2.5, 5])],
    ids=["single_imputation", "multiple_imputation"],
)
def test_calculate_correct_mean_for_imputation(as_dataframe, ignored_cols, expected):
    """Test whether None as impute_values leads to mean imputation

    Parameters
    ----------
    as_dataframe : bool, optional
        Whether to return the data as pd.DataFrame, by default False (np.array)
    ignored_cols : lisEt
        columns which should be mean imputed
    expected : list
        expected values to impute per column
    """
    X, y = data(as_dataframe)
    ib = fairestimator.IgnoringBiasClassifier(
        skclone(clf), ignored_cols=ignored_cols, impute_values=None
    )

    ib.fit(X, y)

    # the effect of impute_values_ on the generated dataset is covered in the previous
    # test, so we just test the intermediate step
    result = ib.impute_values_
    expected = np.array(expected, dtype=np.float64)
    assert np.array_equal(result, expected)


def test_uncorrect_predictions_regressor():
    X, y = load_iris(return_X_y=True)
    ir = fairestimator.IgnoringBiasRegressor(skclone(regressor), ignored_cols=None)
    ir.fit(X, y)
    regres = skclone(regressor).fit(X, y)
    assert np.array_equal(ir._calculate_uncorrected_predictions(X), regres.predict(X))
