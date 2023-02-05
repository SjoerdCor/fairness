import warnings
import copy
from typing import Iterable

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import scipy.special
import numpy as np
import pandas as pd


class BaseIgnoringBiasEstimator(BaseEstimator):
    _required_parameters = ["estimator"]

    def __init__(
        self,
        estimator,
        ignored_cols: Iterable[int] | None = None,
        impute_values: Iterable | None = None,
        correction_strategy: str = "No",
    ):
        """
        Initialize the estimator

        Parameters
        ----------
        estimator :
            The base estimator
        ignored_cols : Iterable[int] | None, optional
            Indices of the columns which must be ignored by the estimator, by default None
            meaning no columns are ignored
        impute_values : Iterable | None, optional
            None (default) means impute with mean. Iterable with values to impute
            may be provided, must be of same length as `ignored_cols`
        correction_strategy : str, optional
            Imputation can lead to higher average outcomes; via this parameter, the estimator
            can correct for that. By default "No" meaning no correction is performed.
            Regressors allow for "Multiplicative" or "Additive" correction, meaning we
            divide or subtract the average overprediction on the trainset for all
            predictions.
            CLassifiers allow for "Logitadditive" which works like additive but in a
            transformed logit-space, so probabilities are constrained [0, 1]

        """

        self.estimator = estimator
        self.ignored_cols = ignored_cols
        self.impute_values = impute_values
        self.correction_strategy = correction_strategy

    def _more_tags(self):
        """Necessary for scikit-learn compatibility

        This is used in many unit tests. Since `check_estimator` clones the estimator,
        this has to be a class method, and cannot be set to specific instances. It is
        unfortunate we have to test configuration here, but that's how it works.
        """
        if self.ignored_cols:
            # Some test sets have very few columns. Ignoring one can have a big impact
            return {"poor_score": True}
        return {}

    @property
    def n_features_in_(self):
        """Necessary for scikit-learn compatibility"""
        return self.estimator_.n_features_in_

    def _calculate_overprediction(self, X, y):
        """Calculate the overprediction when ignoring bias compared to the true outcomes

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The trainset from which to calculate what the outcomes would be when
            bias is ignored
        y : array_like of shape (n_samples,)
            The actual outcomes

        Raises
        ------
        ValueError
            When `self.correction_strategy is not in allowed list (see __init__)`
        """
        if self.correction_strategy == "No":
            self.overprediction_ = None
            return

        y_pred_mean = self._calculate_uncorrected_predictions_mean(X)
        y_true_mean = self._calculate_true_mean(y)

        if self.correction_strategy == "Additive":
            self.overprediction_ = y_pred_mean - y_true_mean
        elif self.correction_strategy == "Multiplicative":
            self.overprediction_ = y_pred_mean / y_true_mean
        elif self.correction_strategy == "Logitadditive":
            self.overprediction_ = scipy.special.logit(
                y_pred_mean
            ) - scipy.special.logit(y_true_mean)
        else:
            msg = 'Correction strategy must be in ["No", "Additive", Multiplicative", "Logitadditive"]'
            msg += f"not {self.correction_strategy!r}"
            raise ValueError(msg)

    def fit(self, X, y=None):
        """
        Fit estimator and learn how to correct for biases in two ways.

        Learns which impute_values to compute for each column that should be ignored.
        Calculates the amount of overprediction due to the fact that we impute values.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The training input samples
        y : array_Like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
            Fitted estimator

        Raises
        ------
        ValueError
            When `self.correction_strategy is not in allowed list (see __init__)`
            When impute_values is not of same length as ignored_cols
        TypeError
            When self.ignored_cols is not Iterable
        IndexError
            When one of ignored_cols is not a valid index
        """
        X, y = check_X_y(X, y)

        self.ignored_cols_ = copy.copy(self.ignored_cols)
        if self.ignored_cols_ is None:
            self.ignored_cols_ = []
        try:
            iter(self.ignored_cols_)
        except TypeError:
            raise TypeError(
                f"self.ignored_cols must be iterable, not {self.ignored_cols_}"
            )

        if self.impute_values is not None and (
            len(self.ignored_cols) != len(self.impute_values)
        ):
            raise ValueError(
                "self.ignored_cols and self.impute_values must be of equal length."
            )

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.impute_values_ = copy.copy(self.impute_values)
        if self.impute_values_ is None:
            if isinstance(X, pd.DataFrame):
                self.impute_values_ = [X.iloc[:, i].mean() for i in self.ignored_cols_]
            else:
                self.impute_values_ = X[:, self.ignored_cols_].mean(axis=0)

        self._calculate_overprediction(X, y)
        return self

    def _prepare_new_dataset(self, X):
        """Impute values for sensitive attributes

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The set for which the ignored cols must be imputed

        Returns
        -------
        array_like of same shape as input
            The set for which cols are imputed

        Raises
        ------
        ValueError
            When self.ignored_cols_ is not of same length as self.impute_values_
        """
        X_new = np.array(X, dtype=np.float64)
        for i, v in zip(self.ignored_cols_, self.impute_values_):
            X_new[:, i] = v
        return X_new

    def _correct_predictions(self, predictions):
        """Correct predictions by subtracting or dividing the overprediction on the trainset

        Parameters
        ----------
        predictions : array_like of shape (n_samples,)
            The original predictions

        Returns
        -------
        predictions : array_like of shape (n_samples,)
            the corrected predictions

        Raises
        ------
        ValueError
            When correction_strategy is invalid
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
    def _warn_inappropriate_correction_strategy(self):
        if self.correction_strategy == "Logitadditive":
            msg = f"Correction strategy is {self.correction_strategy}, which is only meant for classifiers. "
            msg += 'Consider switching to "Additive" or "Multiplicative".'
            warnings.warn(msg)

    def _calculate_uncorrected_predictions_mean(self, X):
        """Predict without correction for overprediction

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            input features

        Returns
        -------
        y : array_like of shape (n_samples,)
            Predictions
        """
        return self.predict(X, use_correction=False).mean()

    def _calculate_true_mean(self, y):
        """Calculate the average true outcome

        Used in calculating the overprediction

        Parameters
        ----------
        y : array_like 
            The true outcomes

        Returns
        -------
        y_mean: array_like of shape (1,)
            The average value of y
        """
        return y.mean()

    def fit(self, X, y, *args, **kwargs):
        """
        Fit estimator and learn how to correct for biases in two ways.

        Learns which impute_values to compute for each column that should be ignored.
        Calculates the amount of overprediction due to the fact that we impute values.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The training input samples
        y : array_Like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        Raises
        ------
        TypeError
            When base_estimator is not Regressor-like
        """
        if not isinstance(self.estimator, RegressorMixin):
            raise TypeError(
                "Base estimator must be subclass of RegressorMixin for: "
                "IgnoringBiasRegressor. Did you mean to use IgnoringBiasClassifier?"
            )
        self._warn_inappropriate_correction_strategy()
        super().fit(X, y, *args, **kwargs)
        return self

    def predict(self, X, use_correction=True):
        """Predict regression target for X

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The input samples
        use_correction : bool, optional
            Whether to correct for overprediction, by default True

        Returns
        -------
        y
            The predicted values
        """
        check_is_fitted(self)
        check_array(X)

        X_new = self._prepare_new_dataset(X)
        y_pred = self.estimator_.predict(X_new)

        if use_correction:
            self._warn_inappropriate_correction_strategy()
            y_pred = self._correct_predictions(y_pred)
        return y_pred


class IgnoringBiasClassifier(BaseIgnoringBiasEstimator, ClassifierMixin):
    def _more_tags(self):
        """Necessary for scikit-learn compatibility

        This is used in many unit tests. Since `check_estimator` clones the estimator,
        this has to be a class method, and cannot be set to specific instances. It is
        unfortunate we have to test configuration here, but that's how it works.
        """
        specifics = {}
        specifics["binary_only"] = True
        base = super()._more_tags()
        return {**base, **specifics}

    def _warn_inappropriate_correction_strategy(self):
        if self.correction_strategy in [
            "Additive",
            "Multiplicative",
        ]:
            msg = f"Correction strategy is {self.correction_strategy}. "
            msg += "This may lead to probabilities smaller than 0 or larger than 1. "
            msg += 'Consider switching to "Logitadditive"'
            warnings.warn(msg)

    def fit(self, X, y, *args, **kwargs):
        """
        Fit estimator and learn how to correct for biases in two ways.

        Learns which impute_values to compute for each column that should be ignored.
        Calculates the amount of overprediction due to the fact that we impute values.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The training input samples
        y : array_Like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        Raises
        ------
        TypeError
            When base_estimator is not Classifier-like
        ValueError
            When multiclass is determined
        """

        if not isinstance(self.estimator, ClassifierMixin):
            raise TypeError(
                "Base estimator must be subclass of ClassifierMixin for: "
                "IgnoringBiasClassifier. Did you mean to use IgnoringBiasRegressor?"
            )
        self._warn_inappropriate_correction_strategy()
        super().fit(X, y, *args, **kwargs)
        if len(self.classes_) > 2:
            msg = (
                "IgnoringBiasClassifier does not work for multiclass... yet! This can "
                "result in erroneous probabilities and predictions. However, please do"
                " contact the maintainers, who would love to solve this with you!"
            )
            raise NotImplementedError(msg)
        return self

    @property
    def classes_(self):
        """Scikit-learn compatibility"""
        return self.estimator_.classes_

    def _calculate_uncorrected_predictions_mean(self, X):
        """Predict without correction for overprediction

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            input features

        Returns
        -------
        y : array_like of shape (n_samples,)
            Predictions
        """
        return self.predict_proba(X, use_correction=False).mean(axis=0)

    def _calculate_true_mean(self, y):
        """Calculate the average probability of each class

        Parameters
        ----------
        y : array_like of shape (n_samples,)
            True outcomes

        Returns
        -------
        p_avg : array_like of shape (n_classes,)
            The average probability of getting each class
        """
        return np.array([y == c for c in self.classes_]).mean(axis=1)

    def predict(self, X, use_correction=True):
        """Predict class for X

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The input samples
        use_correction : bool, optional
            Whether to correct for overprediction, by default True

        Returns
        -------
        y
            The predicted values
        """
        y_proba = self.predict_proba(X, use_correction)
        return self.classes_[np.argmax(y_proba, axis=1)]

    def predict_proba(self, X, use_correction=True):
        """Predict class probabilities for X

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The input samples
        use_correction : bool, optional
            Whether to correct for overprediction, by default True

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the classes
            corresponds to that in the attribute classes_.
        """
        check_is_fitted(self)
        check_array(X)

        if use_correction:
            self._warn_inappropriate_correction_strategy()

        X_new = self._prepare_new_dataset(X)
        y_pred_proba = self.estimator_.predict_proba(X_new)
        if use_correction:
            y_pred_proba = self._correct_predictions(y_pred_proba)

        return y_pred_proba
