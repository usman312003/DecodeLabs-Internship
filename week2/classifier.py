"""
DecodeLabs Internship - Project 2
Data Classification Using AI (Supervised Learning)

Goal:
Build a basic classification model using a small dataset (Iris),
following the IPO (Input -> Process -> Output) framework:

INPUT   : Load Iris dataset, scale features (StandardScaler)
PROCESS : Train-test split + K-Nearest Neighbors (KNN) algorithm
OUTPUT  : Confusion Matrix + Classification Report (Precision, Recall, F1)

Key Skills demonstrated:
- Data handling (loading & understanding a dataset)
- Supervised learning basics (train/test split, feature scaling)
- Model training & evaluation (KNN, confusion matrix, F1 score)
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def load_data():
    """
    INPUT (Raw Feed):
    Load the Iris dataset - 150 samples, 3 classes (Setosa, Versicolor,
    Virginica), 4 features (sepal length/width, petal length/width).
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    return X, y, target_names


def preprocess(X, y):
    """
    INPUT (Sanitization & Normalization):
    1. Split into training (80%) and testing (20%) sets - shuffled
       to remove order bias.
    2. Scale features using StandardScaler (Mean = 0, Variance = 1)
       - "The Gatekeeper Rule" for distance-based algorithms like KNN.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train, k=5):
    """
    PROCESS (The Logic Skeleton):
    Instantiate, fit, and return a K-Nearest Neighbors classifier.
    K=5 means a new point is classified by the majority vote of its
    5 nearest neighbors (the "Proximity Principle").
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """
    OUTPUT (Response Generation / Feedback Loop):
    Generate predictions and validate them using:
    - Accuracy score
    - Confusion Matrix (TP, FP, FN, TN per class)
    - Classification Report (Precision, Recall, F1 Score)

    Note: "In imbalanced data, accuracy is a lie. We must look deeper" -
    that's why we also report the confusion matrix and F1 score.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names)

    return accuracy, cm, report


def main():
    print("=" * 60)
    print(" DecodeLabs - Project 2: Data Classification Using AI")
    print(" Algorithm: K-Nearest Neighbors (KNN) | Dataset: Iris")
    print("=" * 60)

    # INPUT
    X, y, target_names = load_data()
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(target_names)} classes -> {list(target_names)}")

    # PROCESS (preprocessing + training)
    X_train, X_test, y_train, y_test = preprocess(X, y)
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    model = train_model(X_train, y_train, k=5)

    # OUTPUT
    accuracy, cm, report = evaluate_model(model, X_test, y_test, target_names)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
