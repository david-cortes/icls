# Implicitly Constrained Least Squares

Python implementation of the algorithm for semi-supervised learning (fitting a model when there is both labeled and unlabeled data) described in _Implicitly constrained semi-supervised least squares classification (Krijthe and Loog, 2015)_.

## Model Description
The model consists in assigning labels to the unlabeled data in such a way that, when fitting a model to the whole dataset (labeled + unlabeled + made-up labels), it would result in a lower square loss for the labeled data at training stage. This is achieved by making use of the closed-form solution for least squares:

``` Beta = (X'X)^(-1)X'y```

Where X and y are the labeled and unlabeled examples concatenated. Thus, the optimization problem becomes a QP:

```
argmin_y[nlabeled+1:ntotal] ||X[1:nlabeled](X'X)^(-1)X'y - y[1:nlabeled]||
lb <= y[nlabeled+1:ntotal] <= ub
```

Where, for classification, the labels assigned are constrained to be in ```[0,1]```. The formula can be further expanded to incorporate L2 regularization and case weights as follows:

```argmin_y[nlabeled+1:ntotal] ||X[1:nlabeled](X'WX+LI)^(-1)X'Wy - y[1:nlabeled]||```


Implementation is done through Tensorflow + L-BFGS. Documentation is available internally through docstrings (e.g. you can try ```help(ICLS)```.


## Instalation
Package is available on PyPI, can be installed with

```pip install icls```

## Usage
```python
import pandas as pd, numpy as np
from icls import ICLS

# generating a random, noisy dataset for semi-supervised classification
X = np.random.normal(size=(1000,4))
y = 0.5*X[:,0] - 2*X[:,1] + 0.02*X[:,2]**2
y += np.random.normal(size=y.shape[0])
ybin = 1*(y>=0)
X_known = X[:500,:]
y_known = ybin[:500]
X_unknown = X[500:,:]

# fitting a model
model=ICLS(reg_param=1e-4, add_bias=True, ylim=(0,1))
model.fit(X_known, y_known, X_unknown)

# making prections
model.predict(X)

# fitted coefficients
model.coef_

# fitted labels for the unlabeled data
model.yopt_
```


## Some comments
Note that, as the algorithm is based on the closed-form solution for least squares, which implies solving a large system of linear equations, it doesn’t scale as well as plain linear or logistic regression.

While the algorithm can perform classification, it does so by using squared loss, just like in regression, which on itself can make performance degrade to a larger extent than the improvement achieved by adding unlabeled data. Also the predictions will not be bounded to the ```[0,1]``` interval.

You might also want to try assigning a larger weight to the labeled observations. This can be achieved with the ```weight``` parameter in the fit method. Passing a number there assigns that weight to the labeled cases while the unlabeled ones get weight 1. Try assigning something like ```weight=10``` for better results. You can also pass an array of weights with length matching either the labeled dataset or the full dataset.


## References
* Krijthe, J. H., & Loog, M. (2015, October). Implicitly constrained semi-supervised least squares classification. In International symposium on intelligent data analysis (pp. 158-169). Springer, Cham.
