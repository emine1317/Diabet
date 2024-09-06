# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix,roc_curve, auc,classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# Load and preprocess data
veriler = pd.read_csv('veriler_isleme.csv')
eksik_veri = veriler.iloc[:, 1:6]
df = pd.DataFrame(eksik_veri)
df.replace(0, np.nan, inplace=True)
ortalama = df.mean()
for column in df.columns:
    df[column].fillna(ortalama[column], inplace=True)

doğ = veriler.iloc[:, :1]
sonveriler = pd.concat([doğ, df], axis=1)
sonveriler = pd.concat([sonveriler, veriler.iloc[:, -1:]], axis=1)

X=sonveriler.iloc[:,0:8].values
Y=sonveriler.iloc[:,-1:].values

import statsmodels.api as sm
model = sm.OLS(Y,X).fit()
print(model.summary())

# Train-test split
X = sonveriler.iloc[:, 0:8].values
Y = sonveriler.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# K-NN with GridSearchCV
param_grid = {'n_neighbors': [5, 10, 15, 20], 'metric': ['euclidean', 'manhattan']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3, cv=5)
grid.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid.best_params_)

# Predictions
y_pred = grid.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_scores = grid.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Classification report and metrics
print("Classification Report:\n", classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

sensitivity = recall_score(y_test, y_pred)
print(f'Sensitivity (Recall): {sensitivity}')

specificity = recall_score(y_test, y_pred, pos_label=0)
print(f'Specificity: {specificity}')

kappa = cohen_kappa_score(y_test, y_pred)
print(f'Kappa Katsayısı: {kappa}')

f1 = f1_score(y_test, y_pred)
print(f'F-ölçümü: {f1}')

roc_auc = roc_auc_score(y_test, y_pred)
print(f'AUC Katsayısı: {roc_auc}')

"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm

# Load and preprocess data
veriler = pd.read_csv('veriler_isleme.csv')
eksik_veri = veriler.iloc[:, 1:6]
df = pd.DataFrame(eksik_veri)
df.replace(0, np.nan, inplace=True)
ortalama = df.mean()
for column in df.columns:
    df[column].fillna(ortalama[column], inplace=True)

doğ = veriler.iloc[:, :1]
sonveriler = pd.concat([doğ, df], axis=1)
sonveriler = pd.concat([sonveriler, veriler.iloc[:, -3:]], axis=1)

X = sonveriler.iloc[:, 0:8].values
Y = sonveriler.iloc[:, -1].values

# OLS regression
X_const = sm.add_constant(X)  # Add constant term for OLS regression
model = sm.OLS(Y, X_const).fit()
ols_summary = model.summary2().tables[1]  # Get the summary table

# Print OLS summary
print(ols_summary)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# K-NN with GridSearchCV
param_grid = {'n_neighbors': [5, 10, 15, 20], 'metric': ['euclidean', 'manhattan']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3, cv=5)
grid.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid.best_params_)

# Predictions
y_pred = grid.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_scores = grid.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Classification report and metrics
print("Classification Report:\n", classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

sensitivity = recall_score(y_test, y_pred)
print(f'Sensitivity (Recall): {sensitivity}')

specificity = recall_score(y_test, y_pred, pos_label=0)
print(f'Specificity: {specificity}')

kappa = cohen_kappa_score(y_test, y_pred)
print(f'Kappa Katsayısı: {kappa}')

f1 = f1_score(y_test, y_pred)
print(f'F-ölçümü: {f1}')

roc_auc = roc_auc_score(y_test, y_pred)
print(f'AUC Katsayısı: {roc_auc}')

"""


