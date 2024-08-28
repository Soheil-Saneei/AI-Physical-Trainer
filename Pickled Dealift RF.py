import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns

# Load the data from the Excel file
excel_path = '/Users/soheilsaneei/Desktop/Fitness App/Rep_Count.xlsx'
df = pd.read_excel(excel_path)

# Assume the first column 'class' is the target variable and the rest are features
X = df.drop('class', axis=1)
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the scaler and the model
with open('scaler_deadlift.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
with open('random_forest_model_deadlift.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

# Evaluate the model
y_pred_rf = rf.predict(X_test)
import matplotlib.pyplot as plt

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy of Random Forest: {accuracy_rf:.2f}")

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure()
plot_confusion_matrix(cm_rf, classes=['up', 'down'], title='Confusion Matrix for Random Forest')
plt.show()