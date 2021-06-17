import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# DATASET GENERATION
def generate_bias(series: pd.Series, effect_size: float = 1, power: float = 1) -> pd.Series:
    """
    Calculate bias for sensitive attribute

    Parameters
    ----------
    series : pd.Series
        sensitive attribute for which the bias is calculated.
    effect_size : float, optional
        Size of the bias for 1 std from the mean. The default is 1.
    power : float, optional
        power=1: linear bias, power=2: quadratic bias, etc. The default is 1.

    Returns
    -------
    pd.Series
        DESCRIPTION.

    """
    bias = series.sub(series.mean()).pow(power)
    bias = (bias - bias.mean())/bias.std()  # Make the bias neutral
    return bias * effect_size


def display_df(df: pd.DataFrame, n=10):
    """Nicely display all dataframes with column types
    
    df: the DataFrame to display
    n: the number of lines to show
    """    
    display(df.sample(n, random_state=42)
              .style.format({'Age': "{:.2f}",
                             'Education': "{:.2f}",
                             'SocialSkills': "{:.2f}",
                             'Experience': '{:.2f}',
                             'Gender': "{:d}",
                             'PromotionEligibilitySkill': "{:.2f}",
                             'PromotionEligibilityTrue': "{:.2f}",
                             'SalarySkill': "€{:.2f}",
                             'SalaryTrue': '€{:.2f}'})
        )


# PREDICTION
def predict_series(estimator, X: pd.DataFrame, method='predict') -> pd.Series:
    ''' Return predictions from an estimator as a series with index.

    estimator: sklearn model
    X: the set to be predicted
    method: predict or predict_proba
    '''
    if method == 'predict':
        y_pred = estimator.predict(X)
    elif method == 'predict_proba':
        y_pred = estimator.predict_proba(X)[:, 1]
    else:
        raise ValueError(f'method must be `predict` or `predict_proba`, not {method}')
    return pd.Series(y_pred, index=X.index)

def coefs_to_series(lr, names) -> pd.Series:
    ''' Turn linear regression parameters into series with named index for easy plotting.

    lr: a sklearn linear regression model
    names: list of column names, must be of same length as lr.coef_
    series name: name of the model
    '''
    names = names + ['Intercept']
    values = lr.coef_.tolist() + [lr.intercept_]

    return pd.Series(dict(zip(names, values)))


# CALCULATING FAIRNESS METRICS
def calculate_difference_with_uncertainty(df: pd.DataFrame) -> dict:
    '''Calculate mean and sem of difference between two groups

    df: A pandas dataframe that has two rows and two columns (mean and sem),
    so we can calculate the difference of the means and the sem of the difference
    '''
    assert len(df) == 2, 'Can only calculate difference between 2 groups'
    diff_mean = df.loc[0, 'mean'] - df.loc[1, 'mean']
    diff_sem = df['sem'].pow(2).sum() ** 0.5
    return {'diff_mean': diff_mean, 'diff_sem': diff_sem}

def calculate_disparate_impact(y_pred: pd.Series, sensitive_col: pd.Series) -> dict:
    """Calculate disparate impact with uncertainty

    Disparate impact is the mean difference between two groups

    y_pred: array/pd.Series of predictions
    group: array/pd.Series of group values
    """
    y_per_group = y_pred.groupby(sensitive_col).agg(['mean', 'sem'])
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

def calculate_mutual_information(df: pd.DataFrame, sensitive_col_name: str = 'Age', **kwargs) -> dict:
    """
    Calculate mutual information between all columns of df and the senstive col (also part of df).
    
    Can be a measure of disparate impact or treatment for continuous sensitive columns
    """
    df_copy = df.copy()
    sensitive_col = df_copy.pop(sensitive_col_name)
    dis = mutual_info_regression(df_copy, sensitive_col, **kwargs)
    
    return {c: di for c, di in zip(df_copy.columns, dis)}

def bootstrap(func, df, n_bootstrap: int = 500, **kwargs) -> pd.DataFrame:
    """
    Apply func on n_bootstrap bootstrap samples of df and return dataframe with results.
    """
    result = []
    for _ in range(n_bootstrap):
        bootstrap_sample = df.sample(frac=1, replace=True)
        result.append(func(bootstrap_sample, **kwargs))
    return pd.DataFrame(result)
    
def calculate_bootstrap_confidence_interval(bootstrap_results: pd.DataFrame, ci: float = 0.95):
    """
    Calculate percentile confidence interval from bootstrap results
    """
    quantiles = [0.5 - ci/2, 0.5, 0.5 + ci/2]
    df = (bootstrap_results.quantile(quantiles)
          .transpose().assign(Uncertainty_upper = lambda df: df[quantiles[2]].sub(df[0.5]),
                              Uncertainty_lower = lambda df: df[0.5].sub(df[quantiles[0]]),
                       )
          .transpose()
         )
    return df

# VISUALIZATION
def plot_biases(biases: dict, **kwargs):
    """
    Plot fairness metric with uncertainty for multiple mitigation strategies.

    biases: fairness metrics, dict of dicts structured 
        {"name": {"diff_mean": fairness metric, "diff_sem": uncertainty fairness metric}}
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
    plot_biases(impacts, ax=axes[0], title='Disparate impact')

    treatments = calculate_disparate_treatments(predictions, unbiased_values, group)
    plot_biases(treatments, ax=axes[1], title='Disparate treatment')

    axes[1].set_yticklabels([])
    axes[1].set_ylabel('')
    plt.subplots_adjust(wspace=1e-2)

    plt.tight_layout()
    return axes
