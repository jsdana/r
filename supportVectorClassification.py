import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
letters = pd.read_csv('letter-recognition.txt')

# Split the dataset into features (X) and labels (y)
X = np.array(letters.drop(['letter'], axis=1))
y = np.array(letters['letter'])

# Split the x and y into training (16000) and testing (4000) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
clf = SVC()

# Train the SVM classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
predicted = clf.predict(X_test)

# Calculate overall accuracy
accuracy = clf.score(X_test, y_test)
print("Overall Accuracy:", accuracy)

# Initialize a dictionary to store accuracies for each class
class_accuracies = {}

# Calculate accuracy for each class
for c in np.unique(y_test):
    class_indices = np.where(y_test == c)[0] 
    if len(class_indices) > 0:
        class_predictions = predicted[class_indices]  # Get predictions corresponding to class c
        class_accuracy = (class_predictions == c).mean()  # Calculate accuracy for class c
        class_accuracies[c] = class_accuracy

# Print accuracies for each class
print(f"- Accuracy for class")
for c, accuracy in class_accuracies.items():
    print(f"  class {c}: {accuracy:.4f}")
