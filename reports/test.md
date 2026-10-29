### Argument-Evidence Outline

Goal: Does threshold selection and early stopping help linear models to improve in predicting heart attacks?
Data:
    * Processing pipeline:
        * shuffle+1/50 of the data
        * Removing correlations
        * Onehot encoding (first feature dropped)
        * Adding bias, standarization on the train, applied to the other features
    * Data details:
        * sizes
            fit=(5249, 1349), tune=(656, 1349), valid=(657, 1349)
        * we note it is heavily imbalanced:
            Positive labels: 9.3%
        * stratified split to maintain the labels share in both.

Methodology:
* Recall,Precision, F1
* explain early stopping
* (no need to )

Experiments details:
* picking the step size such that the plot is still quiet smoothly decreasing
    max_iters_logistic = 1000
    step_size_logistic = 0.1
    
    max_iters_sgd = 1000*len(y_fit)
    step_size_sgd = 0.0001
    
    max_iters_gd = 1000
    step_size_gd = 0.01
* mention that those are carried to the regularized models.
* range of the lambda tuning for the linear and logistic regression (10^-6 to 10^2). It is obtained through
* early stopping - 20 for GD, 100 for SGD




Results:
* Scores for the baseline methods, after tuning.

* scores for early stopping for the methods using it.
* scores for threshold selection for the methods using it.
* scores 

Discussions:
*

limitations:
* little data
* for the regularized, we picked with the smallest lambda, whereas the largest could have been more interesting (as the biggest would have corresponded to the least complex model)
Conclusion:
* Threshold - helps.
* Early stopping - in general did not help.






### Paper outline

Outline is:
* title: 'Is Decision Tree better than linear models'
* short intro (very short)
* methodology (biggest)
    * Explain each method briefly
* experiments (explain data processing)
* results
    * Put lots of barcharts
* conclusions (short; some interpretation but that can be short)
    * Yes/No + why

What else:
* documentation (fine)