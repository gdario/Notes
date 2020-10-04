# TO DO

## XGBoost and multi-class or multi-label classification

Write a notebook showing how to work with XGBoost with multi-class and multi-label data. Particular attention should go into:

- Understanding which metric can be used out of the box from XGBoost (it has some multi-label metrics), and which ones instead should be used in combination with the `XGBClassifier` estimator from `sklearn`.

| Type        | Number of Targets | Target Cardinality | Valid Type of Target     |
| ---         | ---               | ---                | ---                      |
| Multiclass  | 1                 | > 2                | `multiclass`             |
| Multilabel  | > 1               | 2 (0 or 1)         | `multilabel-indicator`   |
| Multioutput | > 1               | > 2                | `multiclass-multioutput` |

