# fairness
 An investigation into fairness in AI with IgnoringBiasEstimator.
 
 This repository contains the `IgnoringBiasClassifier` and `IgnoringBiasRegressor` in `fairestimator.py`, an easy-to-implement and succesful meta-estimator that removes the disparate treatment.
 
 As an introduction, there is a three-part blog about the `IgnoringBiasEstimators`:
 
1. [Introducing the IgnoringEstimator](https://github.com/SjoerdCor/fairness/blob/main/blog/1.IntroducingTheIgnoringEstimator.ipynb) introduces measures of fairness and shows how naive approaches do not solve them, and on the other hand shows how easily the IgnoringEstimator is implemented and solves them well
1. [Dealing with more complex biases](https://github.com/SjoerdCor/fairness/blob/main/blog/2.DealingWithMoreComplexBiases.ipynb) first shows how common complex biases are: non-linear, correlated with other attributes and for continuous features, and shows how easy it is to mitigate the disparate treatment with the `IgnoringBiasEstimator`. I also show how little attention there seems to be for this problem in existing approaches.
1. [Ignoring bias for cassification poblems](https://github.com/SjoerdCor/fairness/blob/main/blog/3.IgnoringBiasForClassificationProblems.ipynb) finally shows how to use the Ignoring Estimator for the classic classification problems - since these are more prolific, we can also compare against a wide variety of existing approaches and see the `IgnoringBiasEstimator` does equally well or better both in terms of bias mitigation and accuracy.
