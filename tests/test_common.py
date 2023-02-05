""" General tests for sklearn estimators"""
import sys

import pytest

from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import clone as skclone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.append(r"..\..")
sys.path.append("..")
from fairness import fairestimator

clf = RandomForestClassifier(random_state=42)
regressor = RandomForestRegressor(random_state=42)


@pytest.mark.parametrize(
    "estimator",
    [
        fairestimator.IgnoringBiasClassifier(skclone(clf)),
        fairestimator.IgnoringBiasClassifier(
            skclone(clf), [0], correction_strategy="Logitadditive"
        ),
        fairestimator.IgnoringBiasClassifier(
            skclone(LogisticRegression()),
            [0],
            impute_values=[1],
            correction_strategy="Logitadditive",
        ),
        fairestimator.IgnoringBiasRegressor(skclone(regressor)),
        fairestimator.IgnoringBiasRegressor(
            skclone(LinearRegression()), [0], correction_strategy="Multiplicative"
        ),
        fairestimator.IgnoringBiasRegressor(
            skclone(regressor),
            [0],
            impute_values=[1],
            correction_strategy="Additive",
        ),
    ],
    ids=[
        "EmptyClassifier",
        "IgnoringClassifierLogitAdditive",
        "IgnoringClassifierFixedImputation",
        "EmptyRegressor",
        "IgnoringRegressorMultiplicative",
        "IgnoringRegressorAdditive",
    ],
)
def test_all_estimators(estimator):
    """Test sklearn compatibility"""
    return check_estimator(estimator)
