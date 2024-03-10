import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class NaiveBayesLetterRecognition:
    def fit(self, X, y):
        self.X = X
        self.classes = np.unique(y)
        self.parameters = []
        self.class_features = {c: [] for c in self.classes}
        
        for i, sample in enumerate(X):
            letter = sample[0]  # First element is the letter
            features = sample[1:]  # Remaining elements are features
            self.class_features[y[i]].append(features)
        
        for c in self.classes:
            mean = np.mean(self.class_features[c], axis=0)
            std = np.std(self.class_features[c], axis=0) + 1e-6
            self.parameters.append((mean, std))
        
    def _pdf(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return exponent / (np.sqrt(2 * np.pi) * std)
    
    def predict(self, X):
        predictions = []
        for sample in X:
            posterior = []
            for i, c in enumerate(self.classes):
                prior = len(self.class_features[c]) / len(self.X)
                likelihood = np.sum(np.log(self._pdf(sample[1:], self.parameters[i][0], self.parameters[i][1])))
                posterior.append(np.log(prior) + likelihood)
            predictions.append(self.classes[np.argmax(posterior)])
        return np.array(predictions)

# Load the dataset
data = pd.read_csv('./letter-recognition.txt')

# Split the data into features (X) and labels (y)
X = data.iloc[:, 1:].values  # Features (excluding the first column which contains the letter)
y = data.iloc[:, 0].values   # Labels (first column containing the letter)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes model training
naive_bayes = NaiveBayesLetterRecognition()
naive_bayes.fit(X_train, y_train)

# Make predictions on the test set
predictions = naive_bayes.predict(X_test)

# Calculate overall accuracy
overall_accuracy = (predictions == y_test).mean()
print("Overall Accuracy:", overall_accuracy)

# Initialize a dictionary to store accuracies for each class
class_accuracies = {}

# Calculate accuracy for each class
for c in np.unique(y_test):
    class_indices = np.where(y_test == c)[0]  # Get indices where the true class is c
    if len(class_indices) > 0:
        class_predictions = predictions[class_indices]  # Get predictions corresponding to class c
        class_accuracy = (class_predictions == c).mean()  # Calculate accuracy for class c
        class_accuracies[c] = class_accuracy

# Print accuracies for each class
print(f"- Accuracy for class")
for c, accuracy in class_accuracies.items():
    print(f"  class {c}: {accuracy:.4f}")
