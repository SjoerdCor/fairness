import warnings
import copy

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import scipy.special
import numpy as np
import pandas as pd


class BaseIgnoringBiasEstimator(BaseEstimator):
    _required_parameters = ["estimator"]
    def __init__(
        self, estimator, ignored_cols=None, impute_values=None, correction_strategy="No"
    ):
        """
        estimator: an estimator (classifier or regressor)
        ignored_cols: indices of cols to ignore during predicting
        impute_values: values to use during predicting (by default calculates mean)
                        - must be of same length as ignored_cols
        correction_strategy: how to correct for possible overpredictions, must be in
                            ["No", "Additive", "Multiplicative", "Logitadditive"]

        """
        self.estimator = estimator
        self.ignored_cols = ignored_cols
        self.impute_values = impute_values
        self.correction_strategy = correction_strategy

    def _calculate_overprediction(self, X, y):
        y_pred = self._calculate_uncorrected_predictions(X)
        if self.correction_strategy == "No":
            self.overprediction_ = None
        elif self.correction_strategy == "Additive":
            self.overprediction_ = y_pred.mean() - y.mean()
        elif self.correction_strategy == "Multiplicative":
            self.overprediction_ = y_pred.mean() / y.mean()
        elif self.correction_strategy == "Logitadditive":
            self.overprediction_ = scipy.special.logit(
                y_pred.mean()
            ) - scipy.special.logit(y.mean())
        else:
            msg = 'Correction strategy must be in ["No", "Additive", Multiplicative", "Logitadditive"]'
            msg += f"not {self.correction_strategy}"
            raise ValueError(msg)

    def fit(self, X, y=None):
        """
        Fit estimator and learn how to correct for biases in two ways.

        Learns which values to compute for each column that should be hidden if necessary.
        Calculates the amount of overprediction due to the fact that we impute values.
        """ 
        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.impute_values_ = copy.copy(self.impute_values)
        if self.impute_values_ is None:
            if isinstance(X, pd.DataFrame):
                self.impute_values_ = [X.iloc[:, i].mean() for i in self.ignored_cols]
            elif isinstance(X, np.ndarray):
                self.impute_values_ = X[:, self.ignored_cols].mean(axis=0)
            else:
                raise TypeError(f"X must be a np.array or pd.DataFrame, not {type(X)!r}")

        self._calculate_overprediction(X, y)
        return self

    def _prepare_new_dataset(self, X):
        """
        Impute values for sensitive attributes
        """
        X_new = copy.copy(X)
        ignored_cols = self.ignored_cols or []
        if len(ignored_cols) != len(self.impute_values_):
            raise ValueError(
                "self.ignored_cols and self.impute_values must be of same length."
            )
        for i, v in zip(ignored_cols, self.impute_values_):
            if isinstance(X, pd.DataFrame):
                X_new.iloc[:, i] = v
            elif isinstance(X, np.ndarray):
                X_new[:, i] = v 
            else:
                raise TypeError("X must be a np.array or pd.DataFrame")

            
        return X_new

    def _correct_predictions(self, predictions):
        """
        Correct predictions by subtracting or dividing the overprediction on the trainset
        """
        if self.correction_strategy == "No":
            pass
        elif self.correction_strategy == "Additive":
            predictions -= self.overprediction_
        elif self.correction_strategy == "Multiplicative":
            predictions /= self.overprediction_
        elif self.correction_strategy == "Logitadditive":
            predictions = scipy.special.expit(
                scipy.special.logit(predictions) - self.overprediction_
            )
        else:
            msg = 'Correction strategy must be in ["No", "Additive", Multiplicative", "Logitadditive"]'
            msg += f"not {self.correction_strategy}"
            raise ValueError(msg)
        return predictions


class IgnoringBiasRegressor(BaseIgnoringBiasEstimator, RegressorMixin):
    def _calculate_uncorrected_predictions(self, X):
        return self.predict(X, use_correction=False)

    def predict(self, X, y=None, use_correction=True):
        """Predict new instances."""
        check_is_fitted(self)
        check_array(X)
        if use_correction and self.correction_strategy == "Logitadditive":
            msg = f"Correction strategy is {self.correction_strategy}, which is only meant for classifiers. "
            msg += 'Consider switching to "Additive" or "Multiplicative".'
            warnings.warn(msg)

        X_new = self._prepare_new_dataset(X)
        y_pred = self.estimator_.predict(X_new)

        if use_correction:
            y_pred = self._correct_predictions(y_pred)
        return y_pred


class IgnoringBiasClassifier(BaseIgnoringBiasEstimator, ClassifierMixin):
    @property
    def classes_(self):
        return self.estimator_.classes_

    def _calculate_uncorrected_predictions(self, X):
        # This is really ugly, and should be solved (also to handle multi-output)
        return self.predict_proba(X, use_correction=False)[:, -1]

    def predict(self, X, y=None, use_correction=True):
        """Predict new instances."""
        y_proba = self.predict_proba(X, y, use_correction)
        return self.classes_[np.argmax(y_proba, axis=1)]

    def predict_proba(self, X, y=None, use_correction=True):
        """Predict probability for new instances."""
        check_is_fitted(self)
        check_array(X)

        if use_correction and self.correction_strategy in [
            "Additive",
            "Multiplicative",
        ]:
            msg = f"Correction strategy is {self.correction_strategy}. "
            msg += "This may lead to probabilities smaller than 0 or larger than 1. "
            msg += 'Consider switching to "Logitadditive"'
            warnings.warn(msg)

        X_new = self._prepare_new_dataset(X)
        y_pred_proba = self.estimator_.predict_proba(X_new)
        if use_correction:
            y_pred_proba = self._correct_predictions(y_pred_proba)

        return y_pred_proba
