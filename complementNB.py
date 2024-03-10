import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('./letter-recognition.txt')

# Split the dataset into features (X) and labels (y)
X = data.iloc[:, 1:].values  # Features (excluding the first column which contains the letter)
y = data.iloc[:, 0].values   # Labels (first column containing the letter)

# Split the data into training (16000) and testing (4000) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes model training (Complement Naive Bayes)
naive_bayes_complement = ComplementNB()
naive_bayes_complement.fit(X_train, y_train)

# Make predictions on the test set
predictions_complement = naive_bayes_complement.predict(X_test)

# Calculate overall accuracy
overall_accuracy_complement = (predictions_complement == y_test).mean()
print("Overall Accuracy (Complement Naive Bayes):", overall_accuracy_complement)

# Initialize a dictionary to store accuracies for each class
class_accuracies_complement = {}

# Calculate accuracy for each class
for c in np.unique(y_test):
    class_indices = np.where(y_test == c)[0]
    if len(class_indices) > 0:
        class_predictions = predictions_complement[class_indices]  # Get predictions corresponding to class c
        class_accuracy = (class_predictions == c).mean()  # Calculate accuracy for class c
        class_accuracies_complement[c] = class_accuracy

# Print accuracies for each class
print(f"- Accuracy for class")
for c, accuracy in class_accuracies_complement.items():
    print(f"  class {c}: {accuracy:.4f}")
