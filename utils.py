import pandas as pd
import matplotlib.pyplot as plt


def predict_series(estimator, X: pd.DataFrame) -> pd.Series:
    ''' Return predictions from an estimator as a series with index.
    
    estimator: sklearn model
    X: the set to be predicted
    '''
    return pd.Series(estimator.predict(X), index=X.index)

def coefs_to_series(lr, names) -> pd.Series:
    ''' Turn linear regression parameters into series with named index for easy plotting.
    
    lr: a sklearn linear regression model
    names: list of column names, must be of same length as lr.coef_
    series name: name of the model
    '''
    names = names + ['Intercept']
    values = lr.coef_.tolist() + [lr.intercept_]
    
    return pd.Series({n: coef for n, coef in zip(names, values)})

def calculate_difference_with_uncertainty(df: pd.DataFrame) -> dict:
    '''Calculate mean and sem of difference between two groups
    
    df: A pandas dataframe that has two rows and two columns (mean and sem), so we can calculate the difference of the means and the sem of the difference
    ''' 
    assert len(df) == 2, 'Can only calculate difference between 2 groups'
    diff_mean = df.loc[0, 'mean'] - df.loc[1, 'mean']
    diff_sem = df['sem'].pow(2).sum() ** 0.5
    return {'diff_mean': diff_mean, 'diff_sem': diff_sem}

def calculate_disparate_impact(y_pred, group) -> dict:
    """Calculate disparate impact with uncertainty
    
    Disparate impact is the mean difference between two groups
    
    y_pred: array/pd.Series of predictions
    group: array/pd.Series of group values 
    """
    y_per_group = y_pred.groupby(group).agg(['mean', 'sem'])
    return calculate_difference_with_uncertainty(y_per_group)

def calculate_disparate_impacts(predictions: dict, group: pd.Series) -> dict:
    """
    Calculate disparate impact for multiple sets of predictions
    
    predictions: dict with name of prediction type and predictions as keys and values respectively
    group: Series with the values to which group each prediction belongs
    """
    return {name: calculate_disparate_impact(y_pred, group) for name, y_pred in predictions.items()}

def calculate_disparate_treatment(prediction: pd.Series, unbiased_value: pd.Series, group: pd.Series):
    """
    Calculate disparate treatment with uncertainty
    
    Disparate treatment is the difference between the unbiased value and the predicted value
    
    prediction: Series with predictions
    unbiased_value: Series with unbiased values (not generally known in real world, but can be known for toy datasets)
    group: Series with the values to which group each prediction belongs
    """
    y_err = prediction.sub(unbiased_value)
    err_per_group = y_err.groupby(group).agg(['mean', 'sem'])
    return calculate_difference_with_uncertainty(err_per_group)

def calculate_disparate_treatments(predictions: dict, unbiased_values: pd.Series, gender: pd.Series) -> dict:
    """
    Calculate disparate treatments for multiple sets of predictions
    
    prediction: Series with predictions
    unbiased_value: Series with unbiased values (not generally known in real world, but can be known for toy datasets)
    group: Series with the values to which group each prediction belongs

    """
    return {name: calculate_disparate_treatment(y_pred, unbiased_values, gender) for name, y_pred in predictions.items()}

def plot_biases(biases: dict, **kwargs):
    """
    Plot fairness metric with uncertainty for multiple mitigation strategies.
    
    biases: fairness metrics, dict of dicts structured {"name": {"diff_mean": fairness metric, "diff_sem": uncertainty fairness metric}}
    **kwargs are passed to the DataFrame plot method
    """
    df = pd.DataFrame(biases)
    ax = df.transpose().plot(kind='barh', y='diff_mean', xerr='diff_sem', legend=False, **kwargs)
    ax.invert_yaxis()
    ax.axvline(0, c='k', ls='--')
    ax.set_ylabel('FairnessMethod')
    return ax

def plot_fairness_metrics(predictions: dict, unbiased_values: pd.Series, group: pd.Series):
    """
    Plot disparate impact and disparate treatment in one figure.
    
    prediction: Series with predictions
    unbiased_value: Series with unbiased values (not generally known in real world, but can be known for toy datasets)
    group: Series with the values to which group each prediction belongs
    """
    fig, axes = plt.subplots(1, 2)

    impacts = calculate_disparate_impacts(predictions, group)
    plot_biases(impacts, ax=axes[0], title='Disparate impacts')
    
    treatments = calculate_disparate_treatments(predictions, unbiased_values, group)
    plot_biases(treatments, ax=axes[1], title='Disparate treatment')
    
    axes[1].set_yticklabels([])
    axes[1].set_ylabel('')
    plt.subplots_adjust(wspace=1e-2)

    plt.tight_layout()
    return axes
