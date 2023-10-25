# Ex:06 Feature Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Print the transformed features

# PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/08d911ef-4aea-4c85-8d4f-52d7bea02c5b)
```
df.head()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/e0970cd7-b5e6-4d73-8a9b-ef07272171cc)
```
df.isnull().sum()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/d4eb8f58-654b-4475-b1f2-135569999923)
```
df.info()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/73c92dc1-5cb4-4750-8a7c-a5337ddcfe5d)
```
df.describe()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/b75e862c-3259-4926-b0ab-aaa91fd440ff)
```
df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/488ed508-f76a-41f2-83c7-5e828c6a8a15)
```
sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/cee10d2e-f7e7-4942-b41d-cfd572c39ac7)
```
sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/551c2a1c-ee54-4f6e-ac71-c2901253139b)
```
sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/529b129a-2f1d-4bfc-95e0-547e2c6cd5f8)
```
df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/585e4d18-5ccb-43a0-a535-a1a1b00fdaac)
```
df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/40636df8-3c91-411a-8110-e0c137312b8c)
```
df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/260e488e-abd6-4179-9f6c-b1651d51c9d4)
```
df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/a4a3fdde-b58b-49b7-9d7c-8ea209417a50)
```
from sklearn.preprocessing import PowerTransformer
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/ecb849d2-0157-48fb-8014-dfb253ae6b16)
```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex06/assets/135130074/95f2ccb6-61ef-4d18-be76-2e9e8315a75f)


# RESULT:
Thus feature transformation is done for the given set
