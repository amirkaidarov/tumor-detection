import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# Reading the CSV file:
df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")

# Dropping empty column
df = df.drop("Unnamed: 32", axis=1)

# Dropping unnecessary "id" column
df.drop('id', axis=1, inplace=True)

# Mapping diagnosis column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred)

tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]})
results = pd.DataFrame()
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
dtc_acc = accuracy_score(y_test, y_pred)

tempResults = pd.DataFrame({'Algorithm': ['Decision tree Classifier Method'],
                            'Accuracy': [dtc_acc]})
results = pd.concat([results, tempResults])
results = results[['Algorithm', 'Accuracy']]

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
rfc_acc = accuracy_score(y_test, y_pred)

tempResults = pd.DataFrame({'Algorithm': ['Random Forest Classifier Method'],
                            'Accuracy': [rfc_acc]})
results = pd.concat([results, tempResults])
results = results[['Algorithm', 'Accuracy']]

# Support Vector Machine
svc = svm.SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, y_pred)

tempResults = pd.DataFrame({'Algorithm': ['Support Vector Classifier Method'],
                            'Accuracy': [svc_acc]})
results = pd.concat([results, tempResults])
results = results[['Algorithm', 'Accuracy']]

print(results)