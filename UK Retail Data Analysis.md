```python
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error

# Import specific modules from each package
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from prophet import Prophet
from sklearn.model_selection import train_test_split

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import prophet
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric

import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
```


```python
# Read Excel file
df1 = pd.read_excel('/Users/duyixuan/Downloads/online_retail_II.xlsx',sheet_name='Year 2009-2010')
df2 = pd.read_excel('/Users/duyixuan/Downloads/online_retail_II.xlsx',sheet_name='Year 2010-2011')
df = pd.concat([df1,df2])
```

### Exploratory Data Analysis


```python
# Overview
df.head(10)
```




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
      <th>Invoice</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>Price</th>
      <th>Customer ID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>489434</td>
      <td>85048</td>
      <td>15CM CHRISTMAS GLASS BALL 20 LIGHTS</td>
      <td>12</td>
      <td>2009-12-01 07:45:00</td>
      <td>6.95</td>
      <td>13085.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>489434</td>
      <td>79323P</td>
      <td>PINK CHERRY LIGHTS</td>
      <td>12</td>
      <td>2009-12-01 07:45:00</td>
      <td>6.75</td>
      <td>13085.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>489434</td>
      <td>79323W</td>
      <td>WHITE CHERRY LIGHTS</td>
      <td>12</td>
      <td>2009-12-01 07:45:00</td>
      <td>6.75</td>
      <td>13085.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>489434</td>
      <td>22041</td>
      <td>RECORD FRAME 7" SINGLE SIZE</td>
      <td>48</td>
      <td>2009-12-01 07:45:00</td>
      <td>2.10</td>
      <td>13085.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>489434</td>
      <td>21232</td>
      <td>STRAWBERRY CERAMIC TRINKET BOX</td>
      <td>24</td>
      <td>2009-12-01 07:45:00</td>
      <td>1.25</td>
      <td>13085.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>5</th>
      <td>489434</td>
      <td>22064</td>
      <td>PINK DOUGHNUT TRINKET POT</td>
      <td>24</td>
      <td>2009-12-01 07:45:00</td>
      <td>1.65</td>
      <td>13085.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>6</th>
      <td>489434</td>
      <td>21871</td>
      <td>SAVE THE PLANET MUG</td>
      <td>24</td>
      <td>2009-12-01 07:45:00</td>
      <td>1.25</td>
      <td>13085.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>7</th>
      <td>489434</td>
      <td>21523</td>
      <td>FANCY FONT HOME SWEET HOME DOORMAT</td>
      <td>10</td>
      <td>2009-12-01 07:45:00</td>
      <td>5.95</td>
      <td>13085.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>8</th>
      <td>489435</td>
      <td>22350</td>
      <td>CAT BOWL</td>
      <td>12</td>
      <td>2009-12-01 07:46:00</td>
      <td>2.55</td>
      <td>13085.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>9</th>
      <td>489435</td>
      <td>22349</td>
      <td>DOG BOWL , CHASING BALL DESIGN</td>
      <td>12</td>
      <td>2009-12-01 07:46:00</td>
      <td>3.75</td>
      <td>13085.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top 10 Country Distribution
df.Country.value_counts()[:10]
```




    Country
    United Kingdom    981330
    EIRE               17866
    Germany            17624
    France             14330
    Netherlands         5140
    Spain               3811
    Switzerland         3189
    Belgium             3123
    Portugal            2620
    Australia           1913
    Name: count, dtype: int64




```python
# Top 10 most selling product
df.groupby(['Description'])[['Quantity']].sum().reset_index().sort_values(by='Quantity',ascending=False)[:10]
```




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
      <th>Description</th>
      <th>Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4402</th>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>57733</td>
    </tr>
    <tr>
      <th>4509</th>
      <td>WORLD WAR 2 GLIDERS ASSTD DESIGNS</td>
      <td>54698</td>
    </tr>
    <tr>
      <th>721</th>
      <td>BROCADE RING PURSE</td>
      <td>47647</td>
    </tr>
    <tr>
      <th>2744</th>
      <td>PACK OF 72 RETRO SPOT CAKE CASES</td>
      <td>46106</td>
    </tr>
    <tr>
      <th>279</th>
      <td>ASSORTED COLOUR BIRD ORNAMENT</td>
      <td>44925</td>
    </tr>
    <tr>
      <th>147</th>
      <td>60 TEATIME FAIRY CAKE CASES</td>
      <td>36326</td>
    </tr>
    <tr>
      <th>2742</th>
      <td>PACK OF 60 PINK PAISLEY CAKE CASES</td>
      <td>31822</td>
    </tr>
    <tr>
      <th>2158</th>
      <td>JUMBO BAG RED RETROSPOT</td>
      <td>30727</td>
    </tr>
    <tr>
      <th>3984</th>
      <td>SMALL POPCORN HOLDER</td>
      <td>29500</td>
    </tr>
    <tr>
      <th>4067</th>
      <td>STRAWBERRY CERAMIC TRINKET BOX</td>
      <td>26563</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top 10 customers
df.groupby(['Customer ID'])[['Quantity']].sum().reset_index().sort_values(by='Quantity',ascending=False)[:10]
```




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
      <th>Customer ID</th>
      <th>Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2430</th>
      <td>14911.0</td>
      <td>99049</td>
    </tr>
    <tr>
      <th>1845</th>
      <td>14298.0</td>
      <td>56495</td>
    </tr>
    <tr>
      <th>705</th>
      <td>13089.0</td>
      <td>32051</td>
    </tr>
    <tr>
      <th>5219</th>
      <td>17841.0</td>
      <td>29846</td>
    </tr>
    <tr>
      <th>698</th>
      <td>13081.0</td>
      <td>29534</td>
    </tr>
    <tr>
      <th>386</th>
      <td>12748.0</td>
      <td>26296</td>
    </tr>
    <tr>
      <th>1711</th>
      <td>14156.0</td>
      <td>25798</td>
    </tr>
    <tr>
      <th>4904</th>
      <td>17511.0</td>
      <td>18939</td>
    </tr>
    <tr>
      <th>5227</th>
      <td>17850.0</td>
      <td>18349</td>
    </tr>
    <tr>
      <th>547</th>
      <td>12921.0</td>
      <td>15359</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

### Data processing and Quality Check


```python
# Add in column for sales calc
df['Sales'] = df.Quantity * df.Price
```


```python
# Check for missing values
missing_values = df.isnull().sum()
# Drop NA
df.dropna(inplace=True)
# Fill NaN values with an empty string 
df['Description'] = df['Description'].fillna('')
```


```python
# Ignore products with price as 0
df = df[df.Price != 0.0]
```


```python
# Turn all description to lower case
df['Description'] = df['Description'].str.lower()
```


```python
# Drop description containing ? damage mouldy adjust amazon mix missing show wrong wet tag test sold as stock bad quality band charge incorrect oop
for item in ['\?','damage','mouldy','adjust','amazon','manual','discount','postage','bad debt','mix','missing','show','wrong','wet','tag','test','sold as','stock','bad quality','bad charge','incorrect','oop']:
    df = df[df['Description'].str.contains(item) == False]
```


```python
# Get rid of outliers
# Calculate the 10th and 90th percentiles
low_threshold = df['Sales'].quantile(0.1)
high_threshold = df['Sales'].quantile(0.9)
# Filter the DataFrame to exclude values outside the 10-90% range
df = df[(df['Sales'] >= low_threshold) & (df['Sales'] <= high_threshold)]
```


```python

```

### customer segmentation


```python
# Group by Customer
NOW = dt.datetime(2011,12,9)
Customer = df.groupby(['Customer ID']).agg({'InvoiceDate': lambda x: (NOW - x.max()).days, # Recency
                                            'Invoice': lambda x: len(x.unique()), # Frequency
                                            'Sales': lambda x: x.sum()})    # Monetary
# Rename
Customer.rename(columns={'InvoiceDate': 'recency', 
                         'Invoice': 'frequency',
                         'Sales': 'monetary_value'}, inplace=True)
```


```python
Customer.head(5)
```




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
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
    </tr>
    <tr>
      <th>Customer ID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <td>528</td>
      <td>2</td>
      <td>169.36</td>
    </tr>
    <tr>
      <th>12347.0</th>
      <td>1</td>
      <td>8</td>
      <td>3246.84</td>
    </tr>
    <tr>
      <th>12348.0</th>
      <td>247</td>
      <td>4</td>
      <td>288.44</td>
    </tr>
    <tr>
      <th>12349.0</th>
      <td>17</td>
      <td>3</td>
      <td>2538.35</td>
    </tr>
    <tr>
      <th>12350.0</th>
      <td>309</td>
      <td>1</td>
      <td>244.60</td>
    </tr>
  </tbody>
</table>
</div>



### K-means Clustering


```python
Customer = Customer.reset_index()
```


```python
# Check skewness
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column])
    plt.show()
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

for item in ['recency','frequency','monetary_value']:
    check_skew(Customer.reset_index(),item)
```

    /var/folders/qw/_vs94csn4_1g7clcggfq5wlr0000gn/T/ipykernel_23778/709766032.py:6: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df_skew[column])



    
![png](output_21_1.png)
    


    /var/folders/qw/_vs94csn4_1g7clcggfq5wlr0000gn/T/ipykernel_23778/709766032.py:6: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df_skew[column])


    recency's: Skew: 0.8857946569538129, : SkewtestResult(statistic=23.584376513127484, pvalue=5.5755614323946105e-123)



    
![png](output_21_4.png)
    


    frequency's: Skew: 13.672418640540855, : SkewtestResult(statistic=88.19398109564673, pvalue=0.0)


    /var/folders/qw/_vs94csn4_1g7clcggfq5wlr0000gn/T/ipykernel_23778/709766032.py:6: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df_skew[column])



    
![png](output_21_7.png)
    


    monetary_value's: Skew: 23.004763309585307, : SkewtestResult(statistic=101.25268483673861, pvalue=0.0)



```python
# Removing Skewness
Customer_df = Customer.apply(lambda x: x.apply(lambda y: y if y == 0 else np.log(y)))
plt.figure(figsize=(9, 9))
plt.subplot(3, 1, 1)
check_skew(Customer_df,'recency')
plt.subplot(3, 1, 2)
check_skew(Customer_df,'frequency')
plt.subplot(3, 1, 3)
check_skew(Customer_df,'monetary_value')
plt.tight_layout()
```

    /var/folders/qw/_vs94csn4_1g7clcggfq5wlr0000gn/T/ipykernel_23778/709766032.py:6: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df_skew[column])



    
![png](output_22_1.png)
    


    recency's: Skew: nan, : SkewtestResult(statistic=nan, pvalue=nan)


    /var/folders/qw/_vs94csn4_1g7clcggfq5wlr0000gn/T/ipykernel_23778/709766032.py:6: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df_skew[column])



    
![png](output_22_4.png)
    


    frequency's: Skew: 0.5991866743247025, : SkewtestResult(statistic=17.060113219247977, pvalue=2.9398487824928043e-65)


    /var/folders/qw/_vs94csn4_1g7clcggfq5wlr0000gn/T/ipykernel_23778/709766032.py:6: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df_skew[column])



    
![png](output_22_7.png)
    


    monetary_value's: Skew: -0.10586358627064527, : SkewtestResult(statistic=-3.242162542958491, pvalue=0.0011862634343521902)



    <Figure size 640x480 with 0 Axes>



```python
# Standardization
scaler = StandardScaler()
Customer_scaled = scaler.fit_transform(Customer_df)
#Customer_df[['recency','frequency','monetary_value'] = scaler.transform(Customer_df[['recency','frequency','monetary_value']])
# fill na
Customer_scaled = np.nan_to_num(Customer_scaled, nan=0)
Customer_df = Customer_df.fillna(0)
#Customer = np.nan_to_num(Customer, nan=0)
```


```python
# Selection optimizaed k
from scipy.spatial.distance import cdist
distortions = [] 
inertias = [] 
mapping1 = {} 
mapping2 = {} 
K = range(1,10) 
  
for k in K: 
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k).fit(Customer_scaled) 
    kmeanModel.fit(Customer_scaled)     
      
    distortions.append(sum(np.min(cdist(Customer_scaled, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / Customer_scaled.shape[0]) 
    inertias.append(kmeanModel.inertia_) 
  
    mapping1[k] = sum(np.min(cdist(Customer_scaled, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / Customer_scaled.shape[0] 
    mapping2[k] = kmeanModel.inertia_
    
# Plot
plt.plot(K,inertias,'bx-')
plt.show()
    
```

    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(



    
![png](output_24_1.png)
    



```python
# Based on the above K = 4
def kmeans(normalised_df_rfm, clusters_number, original_df_rfm):
    
    kmeans = KMeans(n_clusters = clusters_number, random_state = 1)
    kmeans.fit(normalised_df_rfm)
    # Extract cluster labels
    cluster_labels = kmeans.labels_
        
    # Create a cluster label column in original dataset
    df_new = original_df_rfm.assign(Cluster = cluster_labels)
    
    # Initialise TSNE
    model = TSNE(random_state=1)
    transformed = model.fit_transform(df_new)
    
    # Plot t-SNE
    plt.title('Flattened Graph of {} Clusters'.format(clusters_number))
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=cluster_labels, style=cluster_labels, palette="Set1")
    
    return df_new

#plt.figure(figsize=(10, 10))
#kmeans(pd.DataFrame(Customer_scaled,columns=['Customer ID','recency','frequency','monetary_value']), 4, Customer_df)
#plt.tight_layout()
```


```python
customer_kmeans = kmeans(pd.DataFrame(Customer_scaled,columns=['Customer ID','recency','frequency','monetary_value']), 4, Customer_df)
```

    /Users/duyixuan/anaconda3/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(



    
![png](output_26_1.png)
    



```python
# Persona of each clusters
normalised_df_rfm = pd.DataFrame(pd.DataFrame(Customer_scaled,columns=['Customer ID','recency','frequency','monetary_value']), 
                                       index=Customer_df.index, 
                                       columns=Customer_df.columns)
normalised_df_rfm['Cluster'] = customer_kmeans['Cluster']
                                 
# Melt data into long format
df_melt = pd.melt(normalised_df_rfm.reset_index(), 
                    id_vars=['Customer ID', 'Cluster'],
                    value_vars=['recency', 'frequency', 'monetary_value'], 
                    var_name='Metric', 
                    value_name='Value')
plt.xlabel('Metric')
plt.ylabel('Value')
sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')                                 
```




    <Axes: xlabel='Metric', ylabel='Value'>




    
![png](output_27_1.png)
    



```python
# Assigned Customer ID based on Cluster
Segment1 = customer_kmeans[customer_kmeans.Cluster == 0].applymap(np.exp).reset_index()['Customer ID'].unique().tolist()
Segment2 = customer_kmeans[customer_kmeans.Cluster == 1].applymap(np.exp).reset_index()['Customer ID'].unique().tolist()
Segment3 = customer_kmeans[customer_kmeans.Cluster == 2].applymap(np.exp).reset_index()['Customer ID'].unique().tolist()
Segment4 = customer_kmeans[customer_kmeans.Cluster == 3].applymap(np.exp).reset_index()['Customer ID'].unique().tolist()
```


```python

```

### RFM


```python
# RFM score
quantiles = Customer.quantile(q=[0.25,0.5,0.75])
# RFM Segmentation 
Customer_Segment = Customer.copy()
# Assign class based on quantiles
def R_Class(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
def FM_Class(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
# Asign Class
Customer_Segment['R_Quartile'] = Customer_Segment['recency'].apply(R_Class, args=('recency',quantiles,))
Customer_Segment['F_Quartile'] = Customer_Segment['frequency'].apply(FM_Class, args=('frequency',quantiles,))
Customer_Segment['M_Quartile'] = Customer_Segment['monetary_value'].apply(FM_Class, args=('monetary_value',quantiles,))
Customer_Segment['RFMClass'] = Customer_Segment.R_Quartile.map(str) \
                            + Customer_Segment.F_Quartile.map(str) \
                            + Customer_Segment.M_Quartile.map(str)
```


```python
# Derive Customer Lifetime Value
# Assigning weights
recency_weight = 0.4
frequency_weight = 0.3
monetary_weight = 0.3
# Normalizing RFM scores
Customer_Segment['Recency_normalized'] = (Customer_Segment['R_Quartile'] - Customer_Segment['R_Quartile'].min()) / (Customer_Segment['R_Quartile'].max() - Customer_Segment['R_Quartile'].min())
Customer_Segment['Frequency_normalized'] = (Customer_Segment['F_Quartile'] - Customer_Segment['F_Quartile'].min()) / (Customer_Segment['F_Quartile'].max() - Customer_Segment['F_Quartile'].min())
Customer_Segment['Monetary_normalized'] = (Customer_Segment['M_Quartile'] - Customer_Segment['M_Quartile'].min()) / (Customer_Segment['M_Quartile'].max() - Customer_Segment['M_Quartile'].min())
# Calculating CLV
Customer_Segment['CLV'] = (Customer_Segment['Recency_normalized'] * recency_weight +
                     Customer_Segment['Frequency_normalized'] * frequency_weight +
                     Customer_Segment['Monetary_normalized'] * monetary_weight) * 100 / (recency_weight + frequency_weight + monetary_weight)
```


```python
# Display 
Customer_Segment.reset_index()[['Customer ID','RFMClass','CLV']].head(5)
```




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
      <th>Customer ID</th>
      <th>RFMClass</th>
      <th>CLV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12346.0</td>
      <td>121</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12347.0</td>
      <td>444</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12348.0</td>
      <td>232</td>
      <td>43.333333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12349.0</td>
      <td>424</td>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12350.0</td>
      <td>212</td>
      <td>23.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Break down to categories based on Class
Customer_Segment.RFMClass = Customer_Segment.RFMClass.astype(int)
# Divide Customer to four categories based on RFMClass score
Segment1 = Customer_Segment[Customer_Segment.RFMClass >= 300] # top tier
Segment2 = Customer_Segment[(Customer_Segment.RFMClass < 300) & (Customer_Segment.RFMClass >= 200)]
Segment3 = Customer_Segment[Customer_Segment.RFMClass < 200]
```


```python
# Customers in each segmentation
Customer1 = Segment1.reset_index()['Customer ID'].unique().tolist()
Customer2 = Segment2.reset_index()['Customer ID'].unique().tolist()
Customer3 = Segment3.reset_index()['Customer ID'].unique().tolist()
```


```python

```

### Forecast Sales under RFM


```python
List = [Customer1, Customer2, Customer3]
df.InvoiceDate = df.InvoiceDate.dt.date
```


```python
data = df.groupby(['InvoiceDate'])[['Sales']].sum().reset_index()
```


```python
# Plot time series
plt.figure(figsize=(10, 5))
plt.plot(data['InvoiceDate'], data['Sales'])
plt.show()
```


    
![png](output_40_0.png)
    



```python
# Stationary or not
adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(data.Sales.values)
print('ADF test statistic:', adf)
print('ADF p-values:', pval)
print('ADF number of lags used:', usedlag)
print('ADF number of observations:', nobs)
print('ADF critical values:', crit_vals)
print('ADF best information criterion:', icbest)
```

    ADF test statistic: -2.96837196700049
    ADF p-values: 0.037955496185395274
    ADF number of lags used: 11
    ADF number of observations: 592
    ADF critical values: {'1%': -3.441444394224128, '5%': -2.8664345376276454, '10%': -2.569376663737217}
    ADF best information criterion: 11249.098539325361



```python
# Multiplicative or Additive
# Set 'Date' as the index
#data.set_index('InvoiceDate', inplace=True)
retail_sales_train_decompose_result = seasonal_decompose(data['Sales'], model='multiplicative',period=4)
retail_sales_train_decompose_result.plot().show()
retail_sales_train_decompose_multi_resid = retail_sales_train_decompose_result.resid.sum()

retail_sales_train_decompose_result = seasonal_decompose(data['Sales'], model='additive',period=4)
retail_sales_train_decompose_result.plot().show()
retail_sales_train_decompose_add_resid = retail_sales_train_decompose_result.resid.sum()

if retail_sales_train_decompose_multi_resid < retail_sales_train_decompose_add_resid:
     print("Multiplicate  Model")
else:
     print("Additive  Model")
```

    /var/folders/qw/_vs94csn4_1g7clcggfq5wlr0000gn/T/ipykernel_23778/2738576163.py:5: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      retail_sales_train_decompose_result.plot().show()
    /var/folders/qw/_vs94csn4_1g7clcggfq5wlr0000gn/T/ipykernel_23778/2738576163.py:9: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      retail_sales_train_decompose_result.plot().show()


    Multiplicate  Model



    
![png](output_42_2.png)
    



    
![png](output_42_3.png)
    



```python
# Rename columns
df = df.rename(columns={'InvoiceDate':'ds','Sales':'y'})
```


```python
for a in List:
    data = df[df['Customer ID'].isin(a)].groupby(['ds'])[['y']].sum()
    print('---------------------Customer Segment-------------------------')
    
    ## In-sample Forecast
    print('----------In-Sample Forecast-----------')
    # Train Test split
    #X_train, X_test, y_train, y_test = train_test_split(data['ds'],data['y'], test_size=0.2, random_state=42)
    #train_df = pd.concat([X_train, y_train], axis=1)
    #test_df = pd.concat([X_test, y_test], axis=1)
    #test_df = test_df.rename(columns={'X_test':'ds','y_test':'y'})
    #train_df = data.loc[datetime.strptime('2009-12-01', '%Y-%m-%d').date(): datetime.strptime('2011-06-30', '%Y-%m-%d').date()].reset_index()
    #test_df = data.loc[datetime.strptime('2011-07-01', '%Y-%m-%d').date(): datetime.strptime('2011-12-09', '%Y-%m-%d').date()].reset_index()
    n = data.reset_index().shape[0]
    train_df = data.iloc[:int(n*0.8)].reset_index()
    test_df = data.iloc[int(n*0.8):].reset_index()

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True,  seasonality_mode='multiplicative')
    # Add yearly seasonality
    #model.add_seasonality(name='quarterly', period=91.25, fourier_order=10)
    # Tune other hyperparameters
    #model.holidays_prior_scale = 20
    model.fit(train_df)
    
    # Use the model to make a forecast
    forecast = model.predict(test_df)
    # Summarize the forecast
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # plot forecast
    #model.plot(forecast)
    #plt.show()
     # plot expected vs actual
    plt.plot(test_df['y'].values, label='Actual')
    plt.plot(forecast['yhat'].values, label='Predicted')
    plt.legend()
    plt.show()
    
    # Validate & Error
    #sales_cv = cross_validation(model, initial = '100 days', period='90 days', horizon='200 days', parallel='processes')
    #sales_p = performance_metrics(sales_cv)
    #fig = plot_cross_validation_metric(sales_cv, metric='mape')
    #print(sales_p.mean())
    mae = mean_absolute_error(test_df['y'].values, forecast['yhat'].values)
    print('MAE: %.3f' % mae)
    
    ## Out-of-sample Forecast
    print('-----------Out-of-sample Forecas--------------')
    # Create a DataFrame for next 3 months predictions
    future = pd.date_range(start=data.reset_index()['ds'].max(), periods=90) 
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])
    # Use the model to make a forecast
    forecast = model.predict(future)
    # Summarize the forecast
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # plot forecast
    model.plot(forecast)
    plt.show()
```

    15:59:39 - cmdstanpy - INFO - Chain [1] start processing
    15:59:39 - cmdstanpy - INFO - Chain [1] done processing


    ---------------------Customer Segment-------------------------
    ----------In-Sample Forecast-----------
              ds          yhat   yhat_lower    yhat_upper
    0 2011-07-21  10346.395531  7267.009598  13400.335319
    1 2011-07-22   6450.808493  3421.199710   9821.765926
    2 2011-07-24   6291.485505  2993.076137   9581.469070
    3 2011-07-25   8278.145252  5368.587135  11642.852603
    4 2011-07-26   8126.642901  4765.477481  11627.745277



    
![png](output_44_2.png)
    


    MAE: 3511.389
    -----------Out-of-sample Forecas--------------
              ds          yhat    yhat_lower    yhat_upper
    0 2011-12-09  16658.835335  13379.961286  20035.321361
    1 2011-12-10   2770.570698   -291.805121   6098.080683
    2 2011-12-11  15106.952176  12209.343622  18506.652408
    3 2011-12-12  16412.203600  13243.742027  19564.730989
    4 2011-12-13  15433.210440  12276.462575  18672.183609



    
![png](output_44_4.png)
    


    15:59:39 - cmdstanpy - INFO - Chain [1] start processing
    15:59:39 - cmdstanpy - INFO - Chain [1] done processing


    ---------------------Customer Segment-------------------------
    ----------In-Sample Forecast-----------
              ds         yhat   yhat_lower   yhat_upper
    0 2011-05-04  2291.864153   737.441990  3950.149882
    1 2011-05-05  2776.132420  1125.907337  4239.661117
    2 2011-05-06  1698.263981    63.688032  3192.441622
    3 2011-05-08  1758.061106   246.161254  3362.063554
    4 2011-05-09  1835.236901   209.150264  3482.614780



    
![png](output_44_7.png)
    


    MAE: 749.107
    -----------Out-of-sample Forecas--------------
              ds         yhat  yhat_lower   yhat_upper
    0 2011-09-04   895.913634 -655.990677  2487.403545
    1 2011-09-05   992.207683 -656.897756  2749.863669
    2 2011-09-06  1047.703685 -611.749552  2803.736345
    3 2011-09-07  1108.041157 -648.334807  2703.430579
    4 2011-09-08  1515.147855  -39.490949  3137.365810



    
![png](output_44_9.png)
    


    ---------------------Customer Segment-------------------------
    ----------In-Sample Forecast-----------


    15:59:40 - cmdstanpy - INFO - Chain [1] start processing
    15:59:40 - cmdstanpy - INFO - Chain [1] done processing


              ds         yhat   yhat_lower   yhat_upper
    0 2010-09-17  1785.062443   736.508637  2807.922006
    1 2010-09-19  2013.166320  1032.123729  3109.877046
    2 2010-09-20  2281.856421  1261.694317  3340.847814
    3 2010-09-21  2179.098258  1113.449481  3160.860853
    4 2010-09-22  2271.673987  1237.326771  3327.106601



    
![png](output_44_13.png)
    


    MAE: 1194.602
    -----------Out-of-sample Forecas--------------
              ds         yhat  yhat_lower   yhat_upper
    0 2010-11-24  1412.004060  382.353764  2466.905548
    1 2010-11-25  1557.252570  497.899477  2591.266196
    2 2010-11-26  1214.009266   73.746563  2180.021419
    3 2010-11-27   761.434157 -360.962876  1786.962170
    4 2010-11-28  1493.902429  382.003799  2529.206978



    
![png](output_44_15.png)
    



```python

```

### Forecast Sales under k-means


```python
List1 = [Segment1, Segment2, Segment3, Segment4]
```


```python
for a in List1:
    data = df[df['Customer ID'].isin(a)].groupby(['ds'])[['y']].sum()
    print('---------------------Customer Segment-------------------------')
    
    ## In-sample Forecast
    print('----------In-Sample Forecast-----------')
    # Train Test split
    #X_train, X_test, y_train, y_test = train_test_split(data['ds'],data['y'], test_size=0.2, random_state=42)
    #train_df = pd.concat([X_train, y_train], axis=1)
    #test_df = pd.concat([X_test, y_test], axis=1)
    #test_df = test_df.rename(columns={'X_test':'ds','y_test':'y'})
    #train_df = data.loc[datetime.strptime('2009-12-01', '%Y-%m-%d').date(): datetime.strptime('2011-06-30', '%Y-%m-%d').date()].reset_index()
    #test_df = data.loc[datetime.strptime('2011-07-01', '%Y-%m-%d').date(): datetime.strptime('2011-12-09', '%Y-%m-%d').date()].reset_index()
    n = data.reset_index().shape[0]
    train_df = data.iloc[:int(n*0.8)].reset_index()
    test_df = data.iloc[int(n*0.8):].reset_index()

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True,  seasonality_mode='multiplicative')
    # Add yearly seasonality
    #model.add_seasonality(name='quarterly', period=91.25, fourier_order=10)
    # Tune other hyperparameters
    #model.holidays_prior_scale = 20
    model.fit(train_df)
    
    # Use the model to make a forecast
    forecast = model.predict(test_df)
    # Summarize the forecast
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # plot forecast
    #model.plot(forecast)
    #plt.show()
     # plot expected vs actual
    plt.plot(test_df['y'].values, label='Actual')
    plt.plot(forecast['yhat'].values, label='Predicted')
    plt.legend()
    plt.show()
    
    # Validate & Error
    #sales_cv = cross_validation(model, initial = '100 days', period='90 days', horizon='200 days', parallel='processes')
    #sales_p = performance_metrics(sales_cv)
    #fig = plot_cross_validation_metric(sales_cv, metric='mape')
    #print(sales_p.mean())
    mae = mean_absolute_error(test_df['y'].values, forecast['yhat'].values)
    print('MAE: %.3f' % mae)
    
    ## Out-of-sample Forecast
    print('-----------Out-of-sample Forecas--------------')
    # Create a DataFrame for next 3 months predictions
    future = pd.date_range(start=data.reset_index()['ds'].max(), periods=90) 
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])
    # Use the model to make a forecast
    forecast = model.predict(future)
    # Summarize the forecast
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # plot forecast
    model.plot(forecast)
    plt.show()
```

    16:00:39 - cmdstanpy - INFO - Chain [1] start processing
    16:00:39 - cmdstanpy - INFO - Chain [1] done processing


    ---------------------Customer Segment-------------------------
    ----------In-Sample Forecast-----------
              ds        yhat  yhat_lower  yhat_upper
    0 2011-04-14  228.807636   70.571593  380.962272
    1 2011-04-18  207.439342   36.442164  364.556924
    2 2011-04-20  195.296103   39.905159  350.026263
    3 2011-04-21  197.707527   39.338241  354.387641
    4 2011-04-27  148.026935  -15.250170  317.562733



    
![png](output_48_2.png)
    


    MAE: 100.617
    -----------Out-of-sample Forecas--------------
              ds        yhat  yhat_lower  yhat_upper
    0 2011-11-22  143.420183  -21.724353  305.395455
    1 2011-11-23  195.477258   43.731849  348.723121
    2 2011-11-24  196.213823   44.228082  353.910019
    3 2011-11-25  186.746935   35.909669  354.904310
    4 2011-11-26  172.277655   15.023589  321.891327



    
![png](output_48_4.png)
    


    16:00:39 - cmdstanpy - INFO - Chain [1] start processing


    ---------------------Customer Segment-------------------------
    ----------In-Sample Forecast-----------


    16:00:39 - cmdstanpy - INFO - Chain [1] done processing


              ds        yhat  yhat_lower  yhat_upper
    0 2011-08-22  358.630204   43.343137  653.152085
    1 2011-08-23  331.918824   28.100708  668.029101
    2 2011-08-24  310.935126   -6.815679  645.725519
    3 2011-08-25  411.977481  103.428122  717.516173
    4 2011-08-26  273.107168  -52.794257  575.606188



    
![png](output_48_9.png)
    


    MAE: 362.338
    -----------Out-of-sample Forecas--------------
              ds        yhat  yhat_lower   yhat_upper
    0 2011-12-08  861.196911  543.909162  1187.625684
    1 2011-12-09  704.995435  367.829116  1011.459602
    2 2011-12-10  726.707397  399.357042  1043.917096
    3 2011-12-11  731.626407  399.857097  1046.542365
    4 2011-12-12  769.047465  429.649374  1085.997626



    
![png](output_48_11.png)
    


    ---------------------Customer Segment-------------------------
    ----------In-Sample Forecast-----------


    16:00:39 - cmdstanpy - INFO - Chain [1] start processing
    16:00:39 - cmdstanpy - INFO - Chain [1] done processing


              ds        yhat  yhat_lower  yhat_upper
    0 2011-07-27  479.714613  123.542461  878.149154
    1 2011-07-31  428.140448   61.292305  796.915957
    2 2011-08-02  544.172790  185.312614  902.699378
    3 2011-08-03  591.753393  207.963218  985.595610
    4 2011-08-07  505.375400  124.320075  886.031657



    
![png](output_48_15.png)
    


    MAE: 296.723
    -----------Out-of-sample Forecas--------------
              ds        yhat  yhat_lower  yhat_upper
    0 2011-12-04  286.048955  -80.291488  676.807197
    1 2011-12-05  442.232339   35.129502  837.845408
    2 2011-12-06  335.705633  -10.363544  719.124954
    3 2011-12-07  350.068710  -26.681292  706.695247
    4 2011-12-08  326.898902   -5.911285  717.021597



    
![png](output_48_17.png)
    


    ---------------------Customer Segment-------------------------
    ----------In-Sample Forecast-----------


    16:00:40 - cmdstanpy - INFO - Chain [1] start processing
    16:00:40 - cmdstanpy - INFO - Chain [1] done processing


              ds        yhat  yhat_lower   yhat_upper
    0 2011-07-29  535.699075 -219.186261  1348.158279
    1 2011-07-31  470.233295 -280.545241  1165.787329
    2 2011-08-02  539.556423 -212.189438  1286.264289
    3 2011-08-03  571.887108 -200.797903  1298.508588
    4 2011-08-04  774.411932   15.390049  1541.171988



    
![png](output_48_21.png)
    


    MAE: 537.025
    -----------Out-of-sample Forecas--------------
              ds         yhat  yhat_lower   yhat_upper
    0 2011-12-09  1021.702524  232.576619  1803.896482
    1 2011-12-10  1076.151110  323.242377  1861.596789
    2 2011-12-11   879.494376  152.693948  1623.492671
    3 2011-12-12  1022.246493  237.516134  1783.226540
    4 2011-12-13   875.808455   63.775458  1644.925329



    
![png](output_48_23.png)
    



```python

```
