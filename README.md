# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("titanic_dataset.csv")
df

df.info()

df.describe()

df['Parch'] = df['Parch'].fillna(df['Parch'].dropna().median())
df['Age'] = df['Age'].fillna(df['Age'].dropna().median())
df

df.dropna(how = 'all')

df.dropna(how = 'any')

sns.heatmap(df.isnull())

df.loc[df['Sex']=='male','Sex']=0
df.loc[df['Sex']=='female','Sex']=1
df['Sex']

f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
sns.heatmap(df.isnull(),cbar=False)
df.Survived.value_counts(normalize=True).plot(kind='bar')
plt.show()

cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Outliers removed Dataset")
df.boxplot()
plt.show()

import matplotlib
import seaborn as sns
import numpy as np

import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df.drop("Survived",1) 
y = df["Survived"]          

plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()

cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features

np.asarray(df)

df['Age']=df['Age'].fillna(df['Age'].median())
df

plt.scatter(df.Survived, df.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

df["Age"]=df["Age"].fillna(df["Age"].dropna().median())
df.loc[df['Embarked']=='S','Embarked']=0
df.loc[df['Embarked']=='C','Embarked']=1
df.loc[df['Embarked']=='Q','Embarked']=2
df['Embarked']


```

# OUPUT

![Screenshot (433)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/5ffd15e1-384f-4ff0-ba51-575c29c04ffa)

![Screenshot (434)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/3a37ec15-fe04-478b-980b-c7ae2c99fc6a)

![Screenshot (435)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/57c6fa2e-2249-4a3c-9a03-49858a74a38a)

![Screenshot (436)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/cf892ccf-0564-40c9-af98-5309a14e510f)

![Screenshot (437)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/00499aae-80ba-49af-a8d9-79697deb7931)

![Screenshot (438)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/04d017d1-029e-4d8b-b25a-b84e8bb3d425)

![Screenshot (439)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/7fd6b855-8673-4e9d-9c43-3be9abdb3079)

![Screenshot (440)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/a230cfdc-e2c6-4405-8037-eaa760253063)

![Screenshot (441)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/6ae3b8c7-7720-4246-9e52-567189ae0a48)

![Screenshot (442)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/ae77a42e-a8d7-4ad0-89d1-fbc6ee9038e6)

![Screenshot (443)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/7756214c-68c2-495d-ba78-0f635df6776d)

![Screenshot (444)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/7a885914-4e1b-489c-a383-42aa242a12fa)

![Screenshot (445)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/939efd8d-8a12-42c6-9d6b-254e729189a9)

![Screenshot (446)](https://github.com/VIJAYKUMAR22007124/Ex-07-Feature-Selection/assets/119657657/fefb6e50-3584-4a22-86bb-24c8be8e8a69)

# RESULT 

Thus, Sucessfully performed the various feature selection techniques on a given dataset.


