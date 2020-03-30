---
layout: post
title: "Analyzing Prescribing Behavior Under Medicare Part D: Part 2"
excerpt: "An investigation of the relationship between a care provider's specialty and their prescribing behavior"
categories: [ML, Health]
comments: true
---

In [Part 1](https://llefebure.github.io/articles/2019-06/prescribing-behavior-part1), we did some exploratory analysis on the Roam Analytics Medicare Part D data with an eye towards predicting a provider's specialty from their prescribing behavior. To summarize, the key observations for the classification problem were as follows:

* Highly imbalanced classes
* Widely varying levels of expected separation between classes
* Drug counts follow a power law distribution, so a TFIDF transformation could be useful
* Some features are very sparse
* Strong collinearity in the features

This post (Part 2) focuses on exploring a few models for the multiclass classification task itself.


<div class="input_area" markdown="1">

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics.base
import warnings
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, train_test_split, cross_val_predict)
pd.options.display.max_columns = None
%matplotlib inline
```

</div>


<div class="input_area" markdown="1">

```python
provider_variables = pd.read_csv('data/provider_variables.csv')
vectorizer = np.load('data/vectorizer.npy')[()]
X = np.load('data/X.npy')[()]
```

</div>

## Evaluation Strategy

To compare performance across models, we split the data into separate training and test sets and stratify by specialty to ensure that both partitions have a representative class distribution

We will look at the F1 metric to gauge performance. In the multiclass classifcation case, this metric can take different forms. For example, we can compute the F1 score for each class separately and average the resulting values, or we can sum up the true positive, false positive, etc. counts across all classes and compute a global F1 score. The former implicitly weights each class equally despite their unequal sizes, while the latter favors good performance on the larger classes.

Finally, the F1 metric can be undefined for some classes in the case where there are no predicted values for a class, so we ignore the resulting error.


<div class="input_area" markdown="1">

```python
warnings.filterwarnings(
    'ignore', category=sklearn.exceptions.UndefinedMetricWarning)
```

</div>


<div class="input_area" markdown="1">

```python
provider_variables_train, provider_variables_test, X_train, X_test = \
    train_test_split(
        provider_variables, X, test_size=.2, stratify=provider_variables.specialty)
```

</div>

## Model Building

### Features

First, we transform the features with TFIDF to upweight the rarer and potentially more distinguishing features while downweighting the most popular features.


<div class="input_area" markdown="1">

```python
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train)
X_train_tfidf = tfidf_transformer.transform(X_train)
X_test_tfidf = tfidf_transformer.transform(X_test)
```

</div>

### Logistic Regression

Now that we have features, we build a multinomial logistic regression model. This is a simple model that is easy to train, so it will give us a good baseline level of understanding on the nuances of this problem. The collinearity in the features is problematic if you want to interpret the model coefficients, but it shouldn't affect performance.


<div class="input_area" markdown="1">

```python
lr_model = GridSearchCV(
    LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs'
    ),
    scoring=['f1_macro', 'f1_micro'],
    param_grid={'C': np.logspace(0, 10, 10)},
    return_train_score=True,
    refit='f1_macro'
)
lr_model.fit(X_train_tfidf, provider_variables_train.specialty)
```

</div>




<div class="output_area" markdown="1">

    GridSearchCV(cv=None, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='multinomial',
              n_jobs=1, penalty='l2', random_state=None, solver='lbfgs',
              tol=0.0001, verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'C': array([1.00000e+00, 1.29155e+01, 1.66810e+02, 2.15443e+03, 2.78256e+04,
           3.59381e+05, 4.64159e+06, 5.99484e+07, 7.74264e+08, 1.00000e+10])},
           pre_dispatch='2*n_jobs', refit='f1_macro', return_train_score=True,
           scoring=['f1_macro', 'f1_micro'], verbose=0)

</div>




<div class="input_area" markdown="1">

```python
cv_preds = cross_val_predict(
    lr_model.best_estimator_, X_train_tfidf, provider_variables_train.specialty)
```

</div>

Looking at the individual per class F1 scores, we see a pretty wide range of values. While the larger classes tend to have higher scores, the relationship is not perfect. We also see that the observations from Part 1 that some of the well separated classes such as Neurology and Rheumatology should be fairly easy to predict are indeed true as these classes have an F1 of over .95.

In looking at the confusion matrix of the cross validated predictions, we can definitely see patterns in the errors for the low performing classes. The biggest issue appears to be distinguishing between very similar classes rather than failing completely on strictly small sample classes. For example:

* Child & Adolescent Psychiatry getting mistaken for Psychiatry.
* Acute Care getting mistaken for Family.
* Primary Care getting mistaken for Family.
* Interventional Cardiology getting mistaken for Cardiovascular Disease.


<div class="input_area" markdown="1">

```python
class_labels = provider_variables_train.specialty.value_counts()
f1_scores = f1_score(
    provider_variables_train.specialty, cv_preds, labels=class_labels.index,
    average=None)
class_summary = pd.DataFrame(
    list(zip(class_labels.index, f1_scores, class_labels.values,
             np.argsort(-f1_scores).argsort() + 1)),
    columns=['Specialty', 'F1', 'True Count', 'F1 Rank']
)
class_summary
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Specialty</th>
      <th>F1</th>
      <th>True Count</th>
      <th>F1 Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cardiovascular Disease</td>
      <td>0.898652</td>
      <td>4249</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Family</td>
      <td>0.643547</td>
      <td>3999</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Psychiatry</td>
      <td>0.914622</td>
      <td>2433</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Medical</td>
      <td>0.240791</td>
      <td>1650</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Geriatric Medicine</td>
      <td>0.654655</td>
      <td>1544</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nephrology</td>
      <td>0.931343</td>
      <td>1516</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Neurology</td>
      <td>0.951421</td>
      <td>1393</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Endocrinology, Diabetes &amp; Metabolism</td>
      <td>0.934723</td>
      <td>1116</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Adult Health</td>
      <td>0.137077</td>
      <td>756</td>
      <td>20</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Rheumatology</td>
      <td>0.953229</td>
      <td>683</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Pulmonary Disease</td>
      <td>0.817259</td>
      <td>599</td>
      <td>9</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Infectious Disease</td>
      <td>0.721133</td>
      <td>451</td>
      <td>10</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Adult Medicine</td>
      <td>0.101040</td>
      <td>401</td>
      <td>22</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Hematology &amp; Oncology</td>
      <td>0.850704</td>
      <td>353</td>
      <td>7</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Gastroenterology</td>
      <td>0.817568</td>
      <td>313</td>
      <td>8</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Interventional Cardiology</td>
      <td>0.068241</td>
      <td>312</td>
      <td>23</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Psych/Mental Health</td>
      <td>0.265985</td>
      <td>262</td>
      <td>16</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Gerontology</td>
      <td>0.158491</td>
      <td>158</td>
      <td>19</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Primary Care</td>
      <td>0.010753</td>
      <td>153</td>
      <td>26</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Pain Medicine</td>
      <td>0.596774</td>
      <td>118</td>
      <td>13</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Adolescent Medicine</td>
      <td>0.000000</td>
      <td>118</td>
      <td>27</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Acute Care</td>
      <td>0.017857</td>
      <td>97</td>
      <td>25</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Sports Medicine</td>
      <td>0.047244</td>
      <td>86</td>
      <td>24</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Interventional Pain Medicine</td>
      <td>0.458333</td>
      <td>48</td>
      <td>15</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Child &amp; Adolescent Psychiatry</td>
      <td>0.000000</td>
      <td>43</td>
      <td>28</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Addiction Medicine</td>
      <td>0.109091</td>
      <td>43</td>
      <td>21</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Critical Care Medicine</td>
      <td>0.000000</td>
      <td>42</td>
      <td>29</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Medical Oncology</td>
      <td>0.181818</td>
      <td>42</td>
      <td>18</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Clinical Cardiac Electrophysiology</td>
      <td>0.555556</td>
      <td>40</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>
</div>




<div class="input_area" markdown="1">

```python
confusion_matrix_cv_preds = pd.DataFrame(
    confusion_matrix(provider_variables_train.specialty, cv_preds,
                     labels=class_labels.index))
confusion_matrix_cv_preds.index = class_labels.index
confusion_matrix_cv_preds.columns = class_labels.index
confusion_matrix_cv_preds # element (i,j) refers to number in class i predicted to be in class j
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cardiovascular Disease</th>
      <th>Family</th>
      <th>Psychiatry</th>
      <th>Medical</th>
      <th>Geriatric Medicine</th>
      <th>Nephrology</th>
      <th>Neurology</th>
      <th>Endocrinology, Diabetes &amp; Metabolism</th>
      <th>Adult Health</th>
      <th>Rheumatology</th>
      <th>Pulmonary Disease</th>
      <th>Infectious Disease</th>
      <th>Adult Medicine</th>
      <th>Hematology &amp; Oncology</th>
      <th>Gastroenterology</th>
      <th>Interventional Cardiology</th>
      <th>Psych/Mental Health</th>
      <th>Gerontology</th>
      <th>Primary Care</th>
      <th>Pain Medicine</th>
      <th>Adolescent Medicine</th>
      <th>Acute Care</th>
      <th>Sports Medicine</th>
      <th>Interventional Pain Medicine</th>
      <th>Child &amp; Adolescent Psychiatry</th>
      <th>Addiction Medicine</th>
      <th>Critical Care Medicine</th>
      <th>Medical Oncology</th>
      <th>Clinical Cardiac Electrophysiology</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cardiovascular Disease</th>
      <td>3999</td>
      <td>40</td>
      <td>1</td>
      <td>26</td>
      <td>50</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>14</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>20</td>
      <td>3</td>
      <td>4</td>
      <td>52</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>Family</th>
      <td>59</td>
      <td>2867</td>
      <td>26</td>
      <td>531</td>
      <td>123</td>
      <td>27</td>
      <td>12</td>
      <td>19</td>
      <td>139</td>
      <td>6</td>
      <td>18</td>
      <td>38</td>
      <td>47</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>8</td>
      <td>23</td>
      <td>10</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>9</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Psychiatry</th>
      <td>0</td>
      <td>11</td>
      <td>2330</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>68</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Medical</th>
      <td>38</td>
      <td>942</td>
      <td>11</td>
      <td>353</td>
      <td>65</td>
      <td>9</td>
      <td>23</td>
      <td>15</td>
      <td>62</td>
      <td>8</td>
      <td>10</td>
      <td>30</td>
      <td>35</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>8</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Geriatric Medicine</th>
      <td>45</td>
      <td>144</td>
      <td>2</td>
      <td>47</td>
      <td>1090</td>
      <td>13</td>
      <td>1</td>
      <td>3</td>
      <td>43</td>
      <td>0</td>
      <td>25</td>
      <td>13</td>
      <td>55</td>
      <td>5</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>33</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Nephrology</th>
      <td>18</td>
      <td>22</td>
      <td>0</td>
      <td>11</td>
      <td>25</td>
      <td>1404</td>
      <td>0</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Neurology</th>
      <td>1</td>
      <td>14</td>
      <td>22</td>
      <td>17</td>
      <td>3</td>
      <td>0</td>
      <td>1322</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Endocrinology, Diabetes &amp; Metabolism</th>
      <td>6</td>
      <td>22</td>
      <td>1</td>
      <td>15</td>
      <td>23</td>
      <td>2</td>
      <td>0</td>
      <td>1031</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Adult Health</th>
      <td>46</td>
      <td>364</td>
      <td>8</td>
      <td>85</td>
      <td>60</td>
      <td>14</td>
      <td>4</td>
      <td>13</td>
      <td>83</td>
      <td>1</td>
      <td>7</td>
      <td>22</td>
      <td>12</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Rheumatology</th>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>12</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>642</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Pulmonary Disease</th>
      <td>14</td>
      <td>26</td>
      <td>0</td>
      <td>12</td>
      <td>39</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>483</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Infectious Disease</th>
      <td>6</td>
      <td>26</td>
      <td>0</td>
      <td>12</td>
      <td>29</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>3</td>
      <td>331</td>
      <td>12</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Adult Medicine</th>
      <td>34</td>
      <td>115</td>
      <td>0</td>
      <td>66</td>
      <td>105</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>17</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>34</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Hematology &amp; Oncology</th>
      <td>7</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>302</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Gastroenterology</th>
      <td>12</td>
      <td>16</td>
      <td>1</td>
      <td>7</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>242</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Interventional Cardiology</th>
      <td>295</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Psych/Mental Health</th>
      <td>0</td>
      <td>1</td>
      <td>206</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>52</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Gerontology</th>
      <td>0</td>
      <td>51</td>
      <td>0</td>
      <td>8</td>
      <td>47</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Primary Care</th>
      <td>7</td>
      <td>91</td>
      <td>3</td>
      <td>22</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Pain Medicine</th>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>74</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Adolescent Medicine</th>
      <td>9</td>
      <td>30</td>
      <td>3</td>
      <td>12</td>
      <td>37</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Acute Care</th>
      <td>19</td>
      <td>50</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sports Medicine</th>
      <td>3</td>
      <td>40</td>
      <td>0</td>
      <td>18</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Interventional Pain Medicine</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Child &amp; Adolescent Psychiatry</th>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Addiction Medicine</th>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>3</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Critical Care Medicine</th>
      <td>10</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Medical Oncology</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Clinical Cardiac Electrophysiology</th>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>
</div>



Finally, we will obtain the F1 scores on the test data.


<div class="input_area" markdown="1">

```python
print('Macro F1: ', f1_score(
    provider_variables_test.specialty, lr_model.predict(X_test_tfidf),
    average='macro'))
print('Micro F1: ', f1_score(
    provider_variables_test.specialty, lr_model.predict(X_test_tfidf),
    average='micro'))
```

</div>

<div class="output_area" markdown="1">

    Macro F1:  0.4571918561271133
    Micro F1:  0.7372719374456995


</div>

### XGBoost

For the sake of having something to compare the logistic regression model against, we train a vanilla XGBoost model with the default parameters. Performance is pretty comparable to that of the Logistic Regression. It underperforms on the Macro F1 and outperforms on the Micro F1, showing that the model is biased towards better performance on the larger classes while ignoring smaller ones.


<div class="input_area" markdown="1">

```python
xgb_model = XGBClassifier()
xgb_model.fit(X_train_tfidf, provider_variables_train.specialty)
```

</div>




<div class="output_area" markdown="1">

    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
           max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
           n_estimators=100, n_jobs=1, nthread=None,
           objective='multi:softprob', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
           subsample=1, verbosity=1)

</div>




<div class="input_area" markdown="1">

```python
preds = model.predict(X_test_tfidf)
```

</div>

<div class="input_area" markdown="1">

```python
print('Macro F1: ', f1_score(
    provider_variables_test.specialty, xgb_model.predict(X_test_tfidf),
    average='macro'))
print('Micro F1: ', f1_score(
    provider_variables_test.specialty, xgb_model.predict(X_test_tfidf),
    average='micro'))
```

</div>

<div class="output_area" markdown="1">

    Macro F1:  0.4193069563936367
    Micro F1:  0.7449174630755866


</div>


## Conclusion

The most interesting finding here for me was that the class imbalance really didn't seem to be the biggest problem, rather the similarity between certain classes was. The model had very good accuracy on small sample specialized domains such as Hematology & Oncology but struggled with the ones that had highly related classes such as Child & Adolescent Psychiatry.

I didn't really do any model tuning here, though if this were a problem worth pursuing further, I think my next steps would be to look more closely at the lower performing classes. For example, if we tried to build a binary classifier for Child & Adolescent Psychiatry vs. Pyschiatry, would there be any distinguishing features between the two?
