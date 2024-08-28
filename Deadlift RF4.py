import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load the dataset
data = pd.read_excel('/Users/soheilsaneei/Desktop/Fitness App/Deadlift_1to4.xlsx')  # Ensure this file contains properly labeled data

# Print column names to debug
print("Columns in the dataset:", data.columns)

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Extract features and labels
if 'class' in data.columns:
    X = data.drop('class', axis=1)  # Features
    y = data['class']  # Labels
else:
    raise KeyError("The column 'class' was not found in the dataset. Please check the column names.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
with open('scaler_deadlift.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define the model
rf = RandomForestClassifier()

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# Best parameters and model
best_rf = random_search.best_estimator_

# Cross-validation scores
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

# Train the model
best_rf.fit(X_train, y_train)

# Save the model
with open('random_forest_model_deadlift.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

# Evaluate the model
accuracy = best_rf.score(X_test, y_test)
print("Test set accuracy:", accuracy)

# Predict the test set
y_pred = best_rf.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, y_pred))

# Feature importance
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]})")

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
