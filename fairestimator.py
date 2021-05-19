import warnings

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import scipy.special
import numpy as np

class BaseIgnoringBiasEstimator(BaseEstimator):
    def __init__(self, estimator, ignored_cols=None, impute_values=None, correction_strategy='No'):
        """        
        estimator: an estimator (classifier or regressor)
        ignored_cols: indices of cols to ignore during predicting
        impute_values: valfues to use during predicting (by default calculates mean) - must be of same length as ignored_cols
        correction_strategy: how to correct for possible overpredictions, must be in ["No", "Additive", "Multiplicative", "Logitadditive"]
        
        """
        self.estimator = estimator
        self.ignored_cols = ignored_cols
        self.impute_values = impute_values
        self.correction_strategy = correction_strategy

    def fit(self, X, y=None):
        """
        Fit estimator and learn how to correct for biases in two ways.
        
        Learns which values to compute for each column that should be hidden if necessary.
        Calculates the amount of overprediction due to the fact that we impute values.
        """
        self.estimator.fit(X, y)
        if self.impute_values is None:
            self.impute_values = [X.iloc[:, i].mean() for i in self.ignored_cols]
        
        y_pred = self.predict(X, use_correction=False)
        if self.correction_strategy == "Additive":
            self.overprediction_ =  y_pred.mean() - y.mean()
        elif self.correction_strategy == 'Multiplicative':
            self.overprediction_ = y_pred.mean() / y.mean()
        elif self.correction_strategy == 'Logitadditive':
            self.overprediction_= scipy.special.logit(y_pred.mean()) - scipy.special.logit(y.mean())
        elif self.correction_strategy != 'No':
            raise ValueError(f'Correction strategy must be in ["No", "Additive", "Multiplicative", "Logitadditive"], not {self.correction_strategy}')

    def _prepare_new_dataset(self, X):
        """
        Impute values for sensitive attributes
        """
        X_new = X.copy()
        ignored_cols = self.ignored_cols or []
        if len(ignored_cols) != len(self.impute_values):
            raise ValueError('self.ignored_cols and self.impute_values must be of same length.')
        for i, v in zip(ignored_cols, self.impute_values):
            X_new.iloc[:, i] = v
        return X_new

    def _correct_predictions(self, predictions):
        """
        Correct predictions by subtracting or dividing the overprediction on the trainset
        """
        if self.correction_strategy == "Additive":
            predictions -= self.overprediction_ 
        elif self.correction_strategy == 'Multiplicative':
            predictions /= self.overprediction_
        elif self.correction_strategy == 'Logitadditive':
            predictions = scipy.special.expit(scipy.special.logit(predictions) - self.overprediction_)
        elif self.correction_strategy != 'No':
            raise ValueError(f'Correction strategy must be in ["No", "Additive", Multiplicative", "Logitadditive"], not {self.correction_strategy}')
        return predictions
    
class IgnoringBiasRegressor(BaseIgnoringBiasEstimator, RegressorMixin):  

    def predict(self, X, y=None, use_correction=True):
        """ Predict new instances."""
        if self.correction_strategy == 'Logitadditive':
            msg = f'Correction strategy is {self.correction_strategy}, which is only meant for classifiers. '
            msg += 'Consider switching to "Additive" or "Multiplicative".'
            warnings.warn(msg)
        
        X_new = self._prepare_new_dataset(X)
        y_pred = self.estimator.predict(X_new)
        
        if use_correction:
            y_pred = self._correct_predictions(y_pred)
        return y_pred
    
class IgnoringBiasClassifier(BaseIgnoringBiasEstimator, ClassifierMixin):
    def predict(self, X, y=None, use_correction=True):
        """Predict new instances."""
        
        y_proba = self.predict_proba(X, y, use_correction)
        return np.argmax(y_proba, axis=1)
    
    def predict_proba(self, X, y=None, use_correction=True):
        """Predict probability for new instances."""
        
        if self.correction_strategy in ["Additive", "Multiplicative"]:
            msg = f'Correction strategy is {self.correction_strategy}. This may lead to probabilities smaller than 0 or larger than 1. '
            msg += 'Consider switching to "Logitadditive"'
            warnings.warn(msg)

        X_new = self._prepare_new_dataset(X)
        y_pred_proba = self.estimator.predict_proba(X_new)
        if use_correction:
            y_pred_proba = self._correct_predictions(y_pred_proba)

        return y_pred_proba
 