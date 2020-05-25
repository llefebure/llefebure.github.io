---
layout: post
title: "Analyzing Prescribing Behavior Under Medicare Part D: Part 1"
excerpt: "An investigation of the relationship between a care provider's specialty and their prescribing behavior"
categories: [ML, Health, Exploratory Analysis]
comments: true
---

In [this blog post](https://roamanalytics.com/2016/09/13/prescription-based-prediction/) from Roam Analytics, the authors delve into data on care providers and their record of prescriptions issued under Medicare Part D. Inspired by this work, this analysis investigates to what extent a care provider's prescribing behavior is predictive of their specialty.

This post (Part 1) introduces the data and dives into some initial exploratory analysis. The next one (Part 2) will focus on the multiclass classification task of predicting a provider's specialty (e.g. Cardiovascular Disease, Pyschiatry, etc) from their counts of prescribed drugs.

## Background and Motivation

Prescribing decisions appear to be a product of a far more nuanced decision making process than I had imagined as someone without a medical background. [This review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5499356/) lays out some of the models and theories of how decisions are made. These models take into account easily quantifiable factors such as demographics and drug prices but also more complex ones such as a provider's "habit persistence". Additionally, the Roam Analytics team summarizes their motivation nicely as follows:

> We expect a doctor's prescribing behavior to be governed by many complex, interacting factors related to her specialty, the place she works, her personal preferences, and so forth. How much of this information is hidden away in the Medicare Part D database?


<div class="input_area" markdown="1">

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.offline as ply
import plotly.graph_objs as go
import scipy.stats
import sklearn.metrics.base
from collections import Counter
from pandas.io.json import json_normalize
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
plt.style.use('fivethirtyeight')
ply.init_notebook_mode(connected=False)
%matplotlib inline
```

</div>


<div markdown="0">
        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v1.48.1
* Copyright 2012-2019, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        
</div>


## Data Preparation

The Medicare Part D data is publicly available from https://data.cms.gov/. In this analysis, however, we use a preprocessed version curated by Roam Analytics and hosted [here](https://www.kaggle.com/roamresearch/prescriptionbasedprediction) on Kaggle. Before progressing any further, it is important to emphasize that this data is from Medicare Part D which is a very particular subset (elderly and disabled people) of all prescriptions written in the United States.

The data is provided in JSONL format, so some light preprocessing is necessary to get this into a usable format for analysis. Each line in the file corresponds to a provider, and associated with each provider are counts of prescriptions and some additional variables, each specified as JSON objects. 


<div class="input_area" markdown="1">

```python
unfiltered_data = pd.read_json(
    './data/roam_prescription_based_prediction.jsonl', lines=True)
unfiltered_data.shape
```

</div>




<div class="output_area" markdown="1">

    (239930, 3)

</div>




<div class="input_area" markdown="1">

```python
pd.DataFrame(unfiltered_data.iloc[0])
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cms_prescription_counts</th>
      <td>{'DOXAZOSIN MESYLATE': 26, 'MIDODRINE HCL': 12...</td>
    </tr>
    <tr>
      <th>npi</th>
      <td>1295763035</td>
    </tr>
    <tr>
      <th>provider_variables</th>
      <td>{'settlement_type': 'non-urban', 'generic_rx_c...</td>
    </tr>
  </tbody>
</table>
</div>
</div>



The authors of the original blog post trimmed the data to avoid sparsity issues. Specifically, they filtered out providers with fewer than 50 unique prescribed drugs. Then, they removed providers with low frequency specialties (fewer than 50 total remaining providers with that specialty). Let's first look at what effect this trimming has.

To do this, we plot the appropriate data distributions as histograms:
1. Number of unique prescribed drugs per provider
2. Number of providers per specialty

We see that both of these distributions are highly skewed. Most providers have a low number of unique prescribed drugs, and most specialties have only a few providers associated with them. The filters described above end up removing nearly 90% of the data. We will use this same filtering method.


<div class="input_area" markdown="1">

```python
unique_drugs_prescribed = unfiltered_data.cms_prescription_counts.apply(
    lambda x: len(x.keys()))
specialty_counts = unfiltered_data.provider_variables.apply(
    lambda x: x['specialty']).value_counts()
plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.hist(np.clip(unique_drugs_prescribed, 0, 100))
plt.xlabel('Number of Unique Drugs (clipped at 100)')
plt.ylabel('Count of Providers')
plt.title('Unique Drugs per Provider')
plt.subplot(1, 2, 2)
plt.hist(np.clip(specialty_counts, 0, 500))
plt.xlabel('Number of Providers (clipped at 500)')
plt.ylabel('Count of Unique Specialties')
plt.title('Number of Providers per Specialty')
plt.show()
```

</div>


![png](/assets/Pyfig/prescribing-behavior-part1_6_0.png)



<div class="input_area" markdown="1">

```python
low_drug_count_mask = (
    unfiltered_data.cms_prescription_counts.apply(lambda x: len(x.keys())) >= 50)
data = unfiltered_data[low_drug_count_mask]
```

</div>


<div class="input_area" markdown="1">

```python
specialty_counts = Counter(data.provider_variables.apply(lambda x: x['specialty']))
specialties_to_ignore = set(
    specialty for specialty, _ in filter(lambda x: x[1] < 50, specialty_counts.items()))
low_frequency_specialty_mask = data.provider_variables.apply(
    lambda x: x['specialty'] not in specialties_to_ignore)
data = data[low_frequency_specialty_mask]
```

</div>

Finally, we expand the data out into two data structures, a data frame for holding the metadata about a provider and a sparse matrix for storing the prescription counts.


<div class="input_area" markdown="1">

```python
provider_variables = json_normalize(data=data.provider_variables)
provider_variables['npi'] = unfiltered_data.npi
provider_variables.to_csv('data/provider_variables.csv')
provider_variables.head()
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
      <th>brand_name_rx_count</th>
      <th>gender</th>
      <th>generic_rx_count</th>
      <th>region</th>
      <th>settlement_type</th>
      <th>specialty</th>
      <th>years_practicing</th>
      <th>npi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>384</td>
      <td>M</td>
      <td>2287</td>
      <td>South</td>
      <td>non-urban</td>
      <td>Nephrology</td>
      <td>7</td>
      <td>1295763035</td>
    </tr>
    <tr>
      <th>1</th>
      <td>316</td>
      <td>M</td>
      <td>1035</td>
      <td>West</td>
      <td>non-urban</td>
      <td>Nephrology</td>
      <td>6</td>
      <td>1992715205</td>
    </tr>
    <tr>
      <th>2</th>
      <td>374</td>
      <td>M</td>
      <td>2452</td>
      <td>Northeast</td>
      <td>urban</td>
      <td>Gastroenterology</td>
      <td>5</td>
      <td>1578587630</td>
    </tr>
    <tr>
      <th>3</th>
      <td>683</td>
      <td>M</td>
      <td>3462</td>
      <td>Midwest</td>
      <td>urban</td>
      <td>Psychiatry</td>
      <td>7</td>
      <td>1932278405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>143</td>
      <td>M</td>
      <td>2300</td>
      <td>Northeast</td>
      <td>urban</td>
      <td>Psychiatry</td>
      <td>7</td>
      <td>1437366804</td>
    </tr>
  </tbody>
</table>
</div>
</div>




<div class="input_area" markdown="1">

```python
vectorizer = DictVectorizer(sparse=True)
X = vectorizer.fit_transform(data.cms_prescription_counts)
np.save('data/X.npy', X)
np.save('data/vectorizer.npy', vectorizer)
```

</div>

## Exploratory Analysis
### Specialties

In the filtered dataset, there are 29 unique specialties represented, and the distribution is highly imbalanced. Cardiovascular Disease accounts for nearly 20% of all of the samples, while the 12 lowest frequency classes account for fewer than 1% each.


<div class="input_area" markdown="1">

```python
plt.figure(figsize=(10,10))
counts = pd.Series(provider_variables.specialty).value_counts()[::-1] 
plt.barh(counts.index, counts.values)
for i, v in enumerate(counts.values):
    plt.text(v, i, '{:.1f}%'.format(100 * v / provider_variables.specialty.shape[0]))
plt.title('Distribution of Specialties')
plt.show()
```

</div>


![png](/assets/Pyfig/prescribing-behavior-part1_13_0.png)


Also, it appears that there are some very similar classes that could be tricky to distinguish. For example, Child & Adolescent Psychiatry versus Psychiatry, and Geriatric Medicine versus Gerontology. To explore these similarities further, we first generate a vector representation for each specialty by summing the prescription counts across all providers with that specialty and then normalizing. Then, we use PCA to project these vectors onto two dimensions.


<div class="input_area" markdown="1">

```python
counts_per_specialty = pd.DataFrame(
    X.todense(), index=provider_variables.specialty
).groupby(
    'specialty'
).sum()
dist_per_specialty = counts_per_specialty.values / \
    counts_per_specialty.values.sum(axis=1).reshape((-1, 1))
```

</div>


<div class="input_area" markdown="1">

```python
pca = PCA(n_components=2)
specialties_pca = pca.fit_transform(dist_per_specialty)
```

</div>

Visualizing the resulting points gives us an idea of the relationships and expected separation between classes. We see clear separation for a few specialized classes such as Neurology and Rheumatology, indicating that these classes will likely be among the easiest to predict accurately.

There are also clusters of similar classes forming in this plot. Psychiatry, Child & Adolescent Psychiatry, and Psych/Mental Health is one example. Medical Oncology and Hematology & Oncology are also close in this space, as are Pain Medicine and Interventional Pain Medicine. The plot is interactive and can be cropped to zoom into regions with dense clusters.


<div class="input_area" markdown="1">

```python
scatter = go.Scattergl(
    x=specialties_pca[:,0],
    y=specialties_pca[:,1],
    mode='markers',
    text=counts_per_specialty.index,
    hoverinfo='text',
    marker=dict(
        opacity=0.5
    )
)
layout = go.Layout(
    title='Decomposition of Empirical Drug Distributions',
    xaxis=dict(
        title='PC1'
    ),
    yaxis=dict(
        title='PC2'
    ),
    hovermode='closest'
)
figure = go.Figure(data=[scatter], layout=layout)
ply.iplot(figure, show_link=False)
```

</div>


<div markdown="0">
<div>
        
  {% raw %}   
  <div id="48d0607c-a396-4f4e-b978-de65a58c5e73" class="plotly-graph-div" style="height:525px; width:100%;"></div>
    <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("48d0607c-a396-4f4e-b978-de65a58c5e73")) {
                    Plotly.newPlot(
                        '48d0607c-a396-4f4e-b978-de65a58c5e73',
                        [{"hoverinfo": "text", "marker": {"opacity": 0.5}, "mode": "markers", "text": ["Acute Care", "Addiction Medicine", "Adolescent Medicine", "Adult Health", "Adult Medicine", "Cardiovascular Disease", "Child & Adolescent Psychiatry", "Clinical Cardiac Electrophysiology", "Critical Care Medicine", "Endocrinology, Diabetes & Metabolism", "Family", "Gastroenterology", "Geriatric Medicine", "Gerontology", "Hematology & Oncology", "Infectious Disease", "Interventional Cardiology", "Interventional Pain Medicine", "Medical", "Medical Oncology", "Nephrology", "Neurology", "Pain Medicine", "Primary Care", "Psych/Mental Health", "Psychiatry", "Pulmonary Disease", "Rheumatology", "Sports Medicine"], "type": "scattergl", "uid": "f4bf45ce-1b05-4cc1-97a9-cd593766638e", "x": [-0.023132527913666556, -0.002920318666197159, -0.027743951264804525, -0.02499269697211784, -0.02733657185761154, -0.0783562674175653, 0.03991876621421361, -0.07133751614500608, -0.029044021427093822, -0.036878742943073126, -0.020116836431006053, -0.029816465165264906, -0.023790758034560613, -0.01863117553739625, 0.012418451582322182, -0.0128966729198855, -0.08554788280310031, 0.19819254524296315, -0.02287869208134485, 0.020049231093260376, -0.04103941201401745, 0.05643464935416761, 0.1684363714512999, -0.0214992187422334, 0.041058615775134776, 0.03991182303155463, -0.02195004735551721, 0.06345971082029275, -0.019970388873746484], "y": [-0.01800118142053705, 0.009236124936841488, -0.012691518855694417, -0.01072388889785657, -0.011774322864405636, -0.028995802781237666, 0.1460098847696736, -0.026134067928752145, -0.011818536389987411, -0.006039980298761454, -0.014874160387846391, -0.008835744979607983, -0.0014405112228489038, -0.007666300101388378, -0.016479143766804986, -0.0009528205195691513, -0.032059005731815804, -0.08650717424024836, -0.015562414914496937, -0.02028830726800406, -0.01677836251230497, 0.03766385351413588, -0.06774863352337838, -0.013966987579973623, 0.14595602471113567, 0.13821388277863947, -0.005924964671688912, -0.025576460621053784, -0.01623947923216334]}],
                        {"hovermode": "closest", "title": {"text": "Decomposition of Empirical Drug Distributions"}, "xaxis": {"title": {"text": "PC1"}}, "yaxis": {"title": {"text": "PC2"}}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('48d0607c-a396-4f4e-b978-de65a58c5e73');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
    </script>
  </div>
  {% endraw %}
</div>


### Prescriptions

The features we will use to predict specialty are the provder's prescription counts. Prescription counts, like words in a text corpus, follow an approximate Zipfian distribution meaning that the frequency each drug is prescribed is inversely proportional to that drug's frequency rank order. This becomes evident when plotting the count of each prescription against its rank in our filtered dataset.


<div class="input_area" markdown="1">

```python
top_n = 500
values = np.array(X.sum(axis=0))[0]
sort_order = np.argsort(values)[::-1][:top_n]
plt.figure(figsize=(10,5))
plt.bar(range(top_n), values[sort_order], width=1)
plt.xlabel('Rank')
plt.ylabel('Prescription Count')
plt.title('Drug Frequency Distribution (top {})'.format(top_n))
plt.show()
```

</div>


![png](/assets/Pyfig/prescribing-behavior-part1_20_0.png)


Taking a look at the top prescribed drugs, we see several related to cardiovascular health (not all that surprising given that the top specialty is Cardiovascular Disease). Lisinopril and Amlodipine are used to treat high blood pressure, and Simvastatin is a cholesterol medication.

The top 15 drugs account for nearly a third of all prescriptions in this data. However, it's important to note that for each provider, the data only contains counts for the drugs that they prescribed more than 10 times in that year (presumably for privacy/anonymity reasons?). I guess that there is an even longer tail that we're missing here because of that.


<div class="input_area" markdown="1">

```python
top_n_drugs = 15
top_drugs_df = pd.DataFrame({
    'Drug Name': np.array(vectorizer.feature_names_)[sort_order[:top_n_drugs]],
    'Count': list(map(int, values[sort_order[:top_n_drugs]]))
})
top_drugs_df['Proportion of All Prescriptions'] = top_drugs_df['Count'] / values.sum()
top_drugs_df['Cumulative Proportion of All Prescriptions'] = \
    top_drugs_df['Proportion of All Prescriptions'].cumsum()
top_drugs_df
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
      <th>Drug Name</th>
      <th>Count</th>
      <th>Proportion of All Prescriptions</th>
      <th>Cumulative Proportion of All Prescriptions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LISINOPRIL</td>
      <td>3974707</td>
      <td>0.028034</td>
      <td>0.028034</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AMLODIPINE BESYLATE</td>
      <td>3973254</td>
      <td>0.028024</td>
      <td>0.056058</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SIMVASTATIN</td>
      <td>3712310</td>
      <td>0.026183</td>
      <td>0.082241</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FUROSEMIDE</td>
      <td>3381013</td>
      <td>0.023847</td>
      <td>0.106088</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LEVOTHYROXINE SODIUM</td>
      <td>3295257</td>
      <td>0.023242</td>
      <td>0.129329</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ATORVASTATIN CALCIUM</td>
      <td>3278990</td>
      <td>0.023127</td>
      <td>0.152456</td>
    </tr>
    <tr>
      <th>6</th>
      <td>METOPROLOL TARTRATE</td>
      <td>2728087</td>
      <td>0.019241</td>
      <td>0.171698</td>
    </tr>
    <tr>
      <th>7</th>
      <td>OMEPRAZOLE</td>
      <td>2640322</td>
      <td>0.018622</td>
      <td>0.190320</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HYDROCODONE-ACETAMINOPHEN</td>
      <td>2526317</td>
      <td>0.017818</td>
      <td>0.208139</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CLOPIDOGREL</td>
      <td>2455973</td>
      <td>0.017322</td>
      <td>0.225461</td>
    </tr>
    <tr>
      <th>10</th>
      <td>METOPROLOL SUCCINATE</td>
      <td>2368651</td>
      <td>0.016706</td>
      <td>0.242167</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CARVEDILOL</td>
      <td>2282466</td>
      <td>0.016098</td>
      <td>0.258266</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GABAPENTIN</td>
      <td>2241780</td>
      <td>0.015811</td>
      <td>0.274077</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WARFARIN SODIUM</td>
      <td>2184686</td>
      <td>0.015409</td>
      <td>0.289486</td>
    </tr>
    <tr>
      <th>14</th>
      <td>METFORMIN HCL</td>
      <td>1898465</td>
      <td>0.013390</td>
      <td>0.302876</td>
    </tr>
  </tbody>
</table>
</div>
</div>



Finally, we look at correlation between the drug features. There appear to be several drugs prescribed by only one provider which causes perfect correlation between some features. Looking at the distribution of unique providers for each drug, we see that there are quite a few drugs that have a limited number of providers.

Excluding those drugs that have been prescribed by only one provider, we compute the correlation coefficient between each pair and sort. The top of the list shows a lot of HIV drugs with very high correlations.


<div class="input_area" markdown="1">

```python
drug_correlations = np.corrcoef(X.todense(), rowvar=False)
np.fill_diagonal(drug_correlations, 0)
drug_correlations = np.triu(drug_correlations)
```

</div>


<div class="input_area" markdown="1">

```python
unique_provider_counts = np.array((X > 0).sum(axis=0))[0]
plt.figure(figsize=(10,5))
clip_at = 200
pd.Series(np.clip(unique_provider_counts, 0, clip_at)).plot(
    kind='hist', title='Drugs by Number of Unique Providers (clipped at {})'.format(clip_at))
plt.show()
```

</div>


![png](/assets/Pyfig/prescribing-behavior-part1_25_0.png)



<div class="input_area" markdown="1">

```python
correlated_pairs = zip(*(drug_correlations > .85).nonzero())
pd.DataFrame(
    sorted([
        (vectorizer.feature_names_[i], vectorizer.feature_names_[j], drug_correlations[i, j],
         unique_provider_counts[i], unique_provider_counts[j])
        for i, j in correlated_pairs
        if unique_provider_counts[i] > 1 and unique_provider_counts[j] > 1],
        key=lambda x: x[2], reverse=True),
    columns=['Drug 1', 'Drug 2', 'Correlation', 'Drug 1 Provider Count', 'Drug 2 Provider Count']
)
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
      <th>Drug 1</th>
      <th>Drug 2</th>
      <th>Correlation</th>
      <th>Drug 1 Provider Count</th>
      <th>Drug 2 Provider Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NORVIR</td>
      <td>PREZISTA</td>
      <td>0.962253</td>
      <td>763</td>
      <td>670</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NORVIR</td>
      <td>TRUVADA</td>
      <td>0.954954</td>
      <td>763</td>
      <td>766</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JENTADUETO</td>
      <td>ULTICARE</td>
      <td>0.930126</td>
      <td>121</td>
      <td>345</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NORVIR</td>
      <td>REYATAZ</td>
      <td>0.924530</td>
      <td>763</td>
      <td>665</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ISENTRESS</td>
      <td>TRUVADA</td>
      <td>0.923778</td>
      <td>694</td>
      <td>766</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PREZISTA</td>
      <td>TRUVADA</td>
      <td>0.923685</td>
      <td>670</td>
      <td>766</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NEPHROCAPS</td>
      <td>TRIPHROCAPS</td>
      <td>0.923031</td>
      <td>12</td>
      <td>31</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PAMIDRONATE DISODIUM</td>
      <td>TRELSTAR</td>
      <td>0.904058</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ISENTRESS</td>
      <td>NORVIR</td>
      <td>0.903457</td>
      <td>694</td>
      <td>763</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ISENTRESS</td>
      <td>PREZISTA</td>
      <td>0.890343</td>
      <td>694</td>
      <td>670</td>
    </tr>
    <tr>
      <th>10</th>
      <td>REYATAZ</td>
      <td>TRUVADA</td>
      <td>0.888508</td>
      <td>665</td>
      <td>766</td>
    </tr>
    <tr>
      <th>11</th>
      <td>EPZICOM</td>
      <td>NORVIR</td>
      <td>0.883219</td>
      <td>595</td>
      <td>763</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SULFASALAZINE DR</td>
      <td>SULFAZINE EC</td>
      <td>0.867608</td>
      <td>307</td>
      <td>249</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FUSILEV</td>
      <td>PAMIDRONATE DISODIUM</td>
      <td>0.866331</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>HYDROXYCHLOROQUINE SULFATE</td>
      <td>METHOTREXATE</td>
      <td>0.865934</td>
      <td>2112</td>
      <td>2147</td>
    </tr>
    <tr>
      <th>15</th>
      <td>RENVELA</td>
      <td>SENSIPAR</td>
      <td>0.865158</td>
      <td>2416</td>
      <td>2530</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ATRIPLA</td>
      <td>NORVIR</td>
      <td>0.864710</td>
      <td>702</td>
      <td>763</td>
    </tr>
    <tr>
      <th>17</th>
      <td>INTELENCE</td>
      <td>ISENTRESS</td>
      <td>0.857914</td>
      <td>507</td>
      <td>694</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ATRIPLA</td>
      <td>TRUVADA</td>
      <td>0.854652</td>
      <td>702</td>
      <td>766</td>
    </tr>
    <tr>
      <th>19</th>
      <td>EPZICOM</td>
      <td>TRUVADA</td>
      <td>0.852499</td>
      <td>595</td>
      <td>766</td>
    </tr>
  </tbody>
</table>
</div>
</div>



## Conclusions

This post details an exploratory analysis of the Medicare Part D data that gets us closer to a better understanding of the relationship between a provider's specialty and their prescription record. To summarize, the key findings related to the prediction task are as follows:

1. Classes are highly imbalanced. The most frequent class accounts for nearly 20% of the data compared to well under 1% for the least frequent class.
2. Classes have varying levels of expected separation. Some specialized ones should be fairly easy to predict accurately.
3. Drug counts follow a power law distribution. TFIDF will be useful to transform features.
4. Some drugs are prescribed only by one or just a handful of providers. Some of these may be worth removing as predictors.
5. There is strong collinearity in the features.

In the next post, we will focus on model building and evaluation.