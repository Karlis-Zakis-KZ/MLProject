import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def score(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, np.round(y_pred))


file_names = file_names = ['LDAP.csv']#, 'MSSQL.csv', 'NETBIOS.csv']#
data_frames = []

df = pd.read_csv('03-11/' + "cicddos2019_dataset.csv", low_memory=False)
data_frames.append(df)
data = pd.concat(data_frames, ignore_index=True)

print(data.columns)

data.drop('Unnamed: 0', axis=1, inplace=True) 

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

categorical_cols = data.select_dtypes(include=['object']).drop('Class', axis=1).columns

le = LabelEncoder()

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

columns_to_drop = ['Class', 'Label']
X = data.drop(columns=columns_to_drop, axis=1)

y = (data['Class'] == 'Attack').astype(int)  # This assumes 'Attack' is the label for attacker rows


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scores = {}

optimizers = [Adam(learning_rate=0.001), SGD(learning_rate=0.001), RMSprop(learning_rate=0.001)]

optimizer_scores = {}
importances = {}

for opt in optimizers:
    num_classes = len(np.unique(y))
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Only one unit for binary classification
    ])

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, validation_split=0.2)
    nn_pred = model.predict(X_test_scaled)
    optimizer_scores[type(opt).__name__] = accuracy_score(y_test, np.round(nn_pred))
    print(optimizer_scores[type(opt).__name__])

    # For each column in the dataset
    for col in X.columns:
        # Make a copy of the test data
        X_test_permuted = X_test_scaled.copy()
        # Permute the column in the copied data
        X_test_permuted[:, X.columns.get_loc(col)] = np.random.permutation(X_test_scaled[:, X.columns.get_loc(col)])
        # Calculate the drop in the score and store it in the importances dictionary
        importances[col] = score(model, X_test_scaled, y_test) - score(model, X_test_permuted, y_test)

    # Print the importances
    for feature, importance in sorted(importances.items(), key=lambda item: item[1], reverse=True):
        print(f"Feature: {feature}, Importance: {importance}")
    
    # Sort the importances
    sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)

    # Get the feature names and importance values
    features = [item[0] for item in sorted_importances]
    importance_values = [item[1] for item in sorted_importances]

    # Create a bar chart
    plt.figure(figsize=(10, 5))
    plt.barh(features, importance_values, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()  # Invert y axis to have the most important feature at the top
    plt.show()

plt.figure(figsize=(10, 6))
plt.bar(optimizer_scores.keys(), optimizer_scores.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Optimizer', fontsize=14)
plt.title('Optimizer Comparison', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()