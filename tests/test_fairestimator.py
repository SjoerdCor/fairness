import sys

import pytest
import numpy as np
import pandas as pd
import scipy.special

from sklearn.base import clone as skclone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris, load_diabetes

sys.path.append(r"..\..")
sys.path.append("..")
from fairness import fairestimator
import importlib

importlib.reload(fairestimator)

clf = RandomForestClassifier(min_samples_leaf=1, max_depth=3, random_state=42)
regressor = RandomForestRegressor(min_samples_leaf=1, max_depth=3, random_state=42)

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


@pytest.mark.parametrize(
    ["estimator", "datasetfunc"],
    [
        (
            fairestimator.IgnoringBiasRegressor(skclone(regressor), ignored_cols=None),
            load_diabetes,
        ),
        (
            fairestimator.IgnoringBiasClassifier(skclone(clf), ignored_cols=None),
            load_iris,
        ),
    ],
    ids=["Regressor", "Classifier"],
)
def test_ignoring_nothing_doesnt_change_predict(estimator, datasetfunc):
    """Test whether setting ignored_cols to None leads to the same results for
    prediction as the underlying estimaotr


    Parameters
    ----------
    estimator : subclass of IgnoringBiasBaseEstimator
        The IgnoringBias estimator
    datasetfunc : callable
        An sklearn function to load the dataset
    """
    X, y = datasetfunc(return_X_y=True)
    estimator.fit(X, y)
    underlying_estimator = skclone(estimator.estimator_).fit(X, y)
    assert np.array_equal(estimator.predict(X), underlying_estimator.predict(X))


def test_calculate_correct_overprediction_without_strategy():
    ib = fairestimator.IgnoringBiasClassifier(
        skclone(clf), ignored_cols=[0], correction_strategy="No"
    )
    X = np.array([[0], [1], [5]])
    y = np.array([0, 0, 1])
    ib.fit(X, y)
    result = ib.overprediction_
    assert result is None


@pytest.mark.parametrize(
    ["correction_strategy", "expected"],
    [("Additive", 0.5), ("Multiplicative", 2 / (3 / 2))],
    ids=["Additive", "Multiplicative"],
)
def test_calculate_correct_overprediction_regression(correction_strategy, expected):
    X = [[0], [1]]
    y = [1, 2]
    ib = fairestimator.IgnoringBiasRegressor(
        LinearRegression(),
        ignored_cols=[0],
        impute_values=[1],
        correction_strategy=correction_strategy,
    )
    ib.fit(X, y)
    assert ib.overprediction_ == expected


def test_calculate_correct_overprediction_classification():
    X = [[0], [1]]
    y = [0, 1]
    lr = LogisticRegression(penalty=None)
    ib = fairestimator.IgnoringBiasClassifier(
        lr, ignored_cols=[0], impute_values=[1], correction_strategy="Logitadditive"
    )
    ib.fit(X, y)

    lr.fit(X, y)
    # predict at imputed value and get probability of positive class
    prediction = lr.predict_proba([[1]])[0, 1]
    assert np.array_equal(
        ib.overprediction_,
        [-scipy.special.logit(prediction), scipy.special.logit(prediction)],
    )  # logit(0.5) = 0; 0.5 is avg prediction


@pytest.mark.parametrize(
    ["correction_strategy", "expected"],
    [("Multiplicative", [0.5, 0.5]), ("Additive", [-1, -1])],
)
def test_use_regression_correction_strategy_correctly(correction_strategy, expected):
    X = [[0], [1]]
    y = [1, 1]
    lr = LinearRegression()
    ib = fairestimator.IgnoringBiasRegressor(
        lr,
        correction_strategy=correction_strategy,
    )
    ib.fit(X, y)
    ib.overprediction_ = 2

    assert np.array_equal(ib.predict(X), expected)


def test_use_logitadditive_correction_strategy_correctly():
    X = [[0], [1]]
    y = [0, 1]
    lr = LogisticRegression(penalty=None)
    ib = fairestimator.IgnoringBiasClassifier(
        lr,
        ignored_cols=[0],
        correction_strategy="Logitadditive",
    )
    ib.fit(X, y)
    ib.overprediction_ = np.array([-1e9, 1e9])

    assert np.array_equal(ib.predict([[-100], [100]]), [0, 0])

    ib.overprediction_ = np.array([1e9, -1e9])
    assert np.array_equal(ib.predict([[-100], [100]]), [1, 1])


# Test that correct error is thrown for not existing correction_strategy


# Add test so that an error is thrown if ignored_cols does not match impute_values
# Add test so that an error is thrown if ignored_cols has an index not in X
# Add test so that an error is thrown when the base_estimator does not match with the type of the IgnoringBiasEstimator
# Add test that predict_proba always adds to 1
