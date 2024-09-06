# -*- coding: utf-8 -*-
'''
#5 kat doğrulama ile çalışan
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout

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
Y = sonveriler.iloc[:, -1:].values

# Standardization
sc = StandardScaler()
X = sc.fit_transform(X)

# Neural Network
def create_model():
    model = Sequential()
    model.add(Dense(64, kernel_initializer='uniform', activation='relu', input_dim=8))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

accuracies = []
for train, test in kfold.split(X, Y):
    model = create_model()
    model.fit(X[train], Y[train], epochs=50, batch_size=32, verbose=0)
    y_pred = model.predict(X[test])
    y_pred = (y_pred > 0.5)
    accuracy = accuracy_score(Y[test], y_pred)
    accuracies.append(accuracy)

# Display results
print("Accuracy for each fold:", accuracies)
print("Mean Accuracy:", np.mean(accuracies))
'''
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, recall_score, f1_score, 
    cohen_kappa_score, roc_auc_score, confusion_matrix, roc_curve, auc
)
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, recall_score, f1_score, 
    cohen_kappa_score, roc_auc_score, confusion_matrix, roc_curve, auc
)
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt

# Load and preprocess data
veriler = pd.read_csv('veriler_isleme.csv')
eksik_veri = veriler.iloc[:, 1:6]
df = pd.DataFrame(eksik_veri)
df.replace(0, np.nan, inplace=True)
ortalama = df.mean()
df.fillna(ortalama, inplace=True)

doğ = veriler.iloc[:, :1]
sonveriler = pd.concat([doğ, df], axis=1)
sonveriler = pd.concat([sonveriler, veriler.iloc[:, -2:]], axis=1)

X = sonveriler.iloc[:, 0:8].values
Y = sonveriler.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Neural Network
def create_model():
    model = Sequential()
    model.add(Dense(128, kernel_initializer='uniform', activation='relu', input_dim=8))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
model = create_model()
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32, verbose=1)

# Predict on test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# ROC Curve
fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Display results
print("Final Classification Report:\n", classification_report(Y_test, y_pred))

accuracy = accuracy_score(Y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

sensitivity = recall_score(Y_test, y_pred)
print(f'Sensitivity (Recall): {sensitivity:.2f}')

specificity = recall_score(Y_test, y_pred, pos_label=0)
print(f'Specificity: {specificity:.2f}')

kappa = cohen_kappa_score(Y_test, y_pred)
print(f'Kappa Score: {kappa:.2f}')

f1 = f1_score(Y_test, y_pred)
print(f'F1 Score: {f1:.2f}')

roc_auc = roc_auc_score(Y_test, y_pred_prob)
print(f'ROC AUC Score: {roc_auc:.2f}')
