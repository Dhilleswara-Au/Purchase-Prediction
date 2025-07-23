import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
DATA_PATH = 'Machine-Learning-A-Z-Codes-Datasets/Machine Learning A-Z/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv'
df = pd.read_csv(DATA_PATH)

# Use only Age and EstimatedSalary as features
X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Save the scaler
joblib.dump(sc, 'scaler.joblib')

# Train models
accuracies = {}
def train_and_save_model(model, name):
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f'{name}.joblib')
    acc = model.score(X_test_scaled, y_test)
    accuracies[name] = acc
    print(f'{name} saved. Accuracy: {acc:.4f}')

# Logistic Regression
lr = LogisticRegression(random_state=0)
train_and_save_model(lr, 'logistic_regression')

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
train_and_save_model(knn, 'knn')

# SVM (linear)
svm = SVC(kernel='linear', random_state=0, probability=True)
train_and_save_model(svm, 'svm')

# Kernel SVM (RBF)
k_svm = SVC(kernel='rbf', random_state=0, probability=True)
train_and_save_model(k_svm, 'kernel_svm')

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)  # GaussianNB ignores scaling, but keep for consistency
joblib.dump(nb, 'naive_bayes.joblib')
acc = nb.score(X_test_scaled, y_test)
accuracies['naive_bayes'] = acc
print(f'naive_bayes saved. Accuracy: {acc:.4f}')

# Decision Tree
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
train_and_save_model(dt, 'decision_tree')

# Random Forest
rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
train_and_save_model(rf, 'random_forest')

# Save accuracies
display_names = {
    'logistic_regression': 'Logistic Regression',
    'knn': 'K-Nearest Neighbors',
    'svm': 'SVM (Linear)',
    'kernel_svm': 'Kernel SVM (RBF)',
    'naive_bayes': 'Naive Bayes',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
}
accuracies_display = {display_names.get(k, k): v for k, v in accuracies.items()}
joblib.dump(accuracies_display, 'accuracies.joblib') 