# Week 2 - Project 2: Data Classification Using AI 📊

## DecodeLabs Industrial Training Kit (Batch 2026)

### Goal
Build a basic classification model using a small dataset (the **Iris**
dataset), following the **IPO Framework** (Input -> Process -> Output)
and the **Supervised Learning** pipeline.

### Key Requirements (from spec)
- [x] **Load and understand a dataset** - Iris dataset (150 samples, 3 classes, 4 features)
- [x] **Split data into training and testing sets** - 80/20 split, shuffled
- [x] **Apply a simple classification algorithm** - K-Nearest Neighbors (KNN, K=5)
- [x] **Feature Scaling** - StandardScaler (Mean = 0, Variance = 1)
- [x] **Output Validation** - Confusion Matrix, Precision, Recall, F1 Score

### Requirements / Setup
This project uses `scikit-learn`. Install it with:
```bash
pip install scikit-learn
```

### How to Run
```bash
python classifier.py
```

### Pipeline (IPO Framework)
| Stage   | What happens |
|---------|----------------------------------------------------------|
| Input   | Load Iris dataset, scale features with `StandardScaler` |
| Process | 80/20 train-test split + `KNeighborsClassifier(n_neighbors=5)` |
| Output  | Accuracy, Confusion Matrix, Classification Report (F1)  |

### Example Output
```
============================================================
 DecodeLabs - Project 2: Data Classification Using AI
 Algorithm: K-Nearest Neighbors (KNN) | Dataset: Iris
============================================================

Dataset loaded: 150 samples, 4 features, 3 classes -> ['setosa', 'versicolor', 'virginica']
Train set: 120 samples | Test set: 30 samples

Accuracy: 93.33%

Confusion Matrix:
[[10  0  0]
 [ 0 10  0]
 [ 0  2  8]]

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.83      1.00      0.91        10
   virginica       1.00      0.80      0.89        10

    accuracy                           0.93        30
   macro avg       0.94      0.93      0.93        30
weighted avg       0.94      0.93      0.93        30
```

### Why Scaling Matters (The Gatekeeper Rule)
KNN is a distance-based algorithm. Without scaling, features with larger
numeric ranges (like petal length in cm) would dominate the distance
calculation over smaller ones. `StandardScaler` puts all features on the
same scale (mean = 0, variance = 1) so each feature contributes fairly.

### Why Accuracy Isn't Enough ("The Accuracy Mirage")
A high accuracy score can hide poor performance on individual classes,
especially with imbalanced data. The **Confusion Matrix** shows exactly
where the model gets confused (e.g., here it confused 2 "virginica"
samples for "versicolor"), and the **F1 Score** balances Precision and
Recall for a more honest picture per class.

### Key Skills Practiced
- Data handling & dataset exploration
- Feature scaling / normalization
- Train-test split with stratification
- Supervised learning with K-Nearest Neighbors
- Model evaluation: Confusion Matrix, Precision, Recall, F1 Score

### Possible Extensions (from the Conclusion section of the kit)
- Try different values of `K` and observe the "elbow" in error rate
- Compare KNN against other algorithms (e.g., Decision Tree, Logistic Regression)
- Test the trained model on completely new/custom flower measurements
