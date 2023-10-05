# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:52:58 2023

@author: imkh3
"""

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
X_path = "C:/Users/user/Desktop/Hackerton/JustArrayData.mat"
y_path = "C:/Users/user/Desktop/Hackerton/JustArrayLabels.mat"

matData = loadmat(X_path)
matLabel = loadmat(y_path)

rawMatData = matData['JustArrayData']
rawMatLabel = matLabel['JustArrayLabels']

rawMatLabel[rawMatLabel == 2] = 0
rawMatLabel[rawMatLabel == 3] = 2

X = np.array(pd.DataFrame(rawMatData))
y = np.array(pd.DataFrame(rawMatLabel)).ravel()

DT_feature = [1, 2, 3, 4, 8, 17, 23, 25, 26, 28, 29, 30, 31, 35, 36]
RF_feature = [1, 2, 4, 6, 7, 8, 10, 13, 15, 17, 20, 22, 23, 25, 28, 29, 30, 34, 35, 38]
MULTI_feature = list(set().union(DT_feature, RF_feature))

DT_X = X[:, DT_feature]
RF_X = X[:, RF_feature]
MULTI_X = X[:, MULTI_feature]

# Initialize 5-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize performance metrics storage
accuracies = []
precisions = []
recalls = []
f1_scores = []
avg_con = np.zeros((3,3))
# Cross-validation loop
for train_index, test_index in kf.split(DT_X):
    X_train, X_test = DT_X[train_index], DT_X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model
    DT = DecisionTreeClassifier(criterion="entropy", max_depth=23)
    DT.fit(X_train, y_train)

    # Predict on test set
    y_pred = DT.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # row_normalized_matrix = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("DT Confusion Matrix:")
    # print(row_normalized_matrix)

    avg_con = avg_con + cm
    # Compute performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

DT_pred = DT.predict(X)
avg_con = avg_con / avg_con.sum(axis=1)[:, np.newaxis]
# Calculate average performance metrics
avg_acc = sum(accuracies) / 10.0
avg_prec = sum(precisions) / 10.0
avg_rec = sum(recalls) / 10.0
avg_f1 = sum(f1_scores) / 10.0

print("DT Average Accuracy:", avg_acc)
print("DT Average Precision:", avg_prec)
print("DT Average Recall:", avg_rec)
print("DT Average F1-Score:", avg_f1)
print("DT Average ConfusionMatrix:")
print(avg_con)
print("")

'''
# Initialize performance metrics storage
accuracies = []
precisions = []
recalls = []
f1_scores = []
avg_con = np.zeros((3,3))
# Cross-validation loop
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model
    RF = RandomForestClassifier(n_estimators=30, criterion="entropy", max_depth=23)
    RF.fit(X_train, y_train)

    # Predict on test set
    y_pred = RF.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    avg_con = avg_con + cm

    # Compute performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

RF_pred = RF.predict(X)
avg_con = avg_con / avg_con.sum(axis=1)[:, np.newaxis]
# Calculate average performance metrics
avg_acc = sum(accuracies)/10.0
avg_prec = sum(precisions)/10.0
avg_rec = sum(recalls)/10.0
avg_f1 = sum(f1_scores)/10.0

print("RF Average Accuracy:", avg_acc)
print("RF Average Precision:", avg_prec)
print("RF Average Recall:", avg_rec)
print("RF Average F1-Score:", avg_f1)
print("RF Average ConfusionMatrix:")
print(avg_con)
print("")
'''
'''
# Initialize performance metrics storage
accuracies = []
precisions = []
recalls = []
f1_scores = []

X_train, X_test, y_train, y_test = train_test_split(DT_X, y, test_size=0.3, random_state=44)

DT = DecisionTreeClassifier(criterion="entropy", max_depth=23)
DT.fit(X_train, y_train)

# Predict on test set
DT_pred = DT.predict(DT_X)
print(accuracy_score(y, DT_pred))
X_train, X_test, y_train, y_test = train_test_split(RF_X, y, test_size=0.3, random_state=44)

RF = RandomForestClassifier(n_estimators=30, criterion="entropy", max_depth=23)
RF.fit(X_train, y_train)

# Predict on test set
RF_pred = RF.predict(RF_X)
print(accuracy_score(y, RF_pred))

multi_X = np.column_stack((DT_pred, RF_pred))


# Cross-validation loop
for train_index, test_index in kf.split(X):

    scaler = StandardScaler()
    scaler.fit(MULTI_X)
    X_scaled = scaler.transform(MULTI_X)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model
    MLP = MLPClassifier(hidden_layer_sizes=(8), tol=0.001, activation='relu', solver='adam', batch_size=32, max_iter=1000)
    MLP.fit(X_train, y_train)

    # Predict on test set
    y_pred = MLP.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("MLP Confusion Matrix:")
    print(cm)

    # Compute performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

# Calculate average performance metrics
avg_acc = sum(accuracies)/10.0
avg_prec = sum(precisions)/10.0
avg_rec = sum(recalls)/10.0
avg_f1 = sum(f1_scores)/10.0

print("MLP Average Accuracy:", avg_acc)
print("MLP Average Precision:", avg_prec)
print("MLP Average Recall:", avg_rec)
print("MLP Average F1-Score:", avg_f1)
'''