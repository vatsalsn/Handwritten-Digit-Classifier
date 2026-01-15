# Handwritten Digits Classifier

## Overview

This project is a **Handwritten Digits Classifier** built using **Scikit-Learn**. It is trained on the well-known **Digits dataset**, which contains images of handwritten numbers (0-9). The goal is to accurately classify each digit using different machine learning models and demonstrate why **Support Vector Machine (SVM) is the most effective model** for this task.

## Features

- **Uses Scikit-Learn's Digits dataset**\
- **Preprocesses data with feature scaling for better performance**\
- **Trains multiple models for comparison**\
- **Evaluates SVM model performance with accuracy and classification reports**\
- **Visualizes sample digits using Matplotlib**\
- **Implements train-test splitting for unbiased model evaluation**

## Installation

Ensure you have **Python 3.x** and install the required dependencies using:

```sh
pip install numpy matplotlib scikit-learn seaborn
```

## Dataset

The dataset used in this project is the **Digits dataset** from Scikit-Learn, consisting of 8×8 grayscale images of handwritten digits (0-9). Each image is represented as a **feature vector of 64 values**.

### Sample Digits Visualization

```python
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.show()
```

## Usage

### Load the Dataset

```python
from sklearn import datasets

digits = datasets.load_digits()
X, y = digits.data, digits.target
print(f"Feature Matrix Shape: {X.shape}")
print(f"Labels Shape: {y.shape}")
```

### Preprocess the Data

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Train Multiple Models for Comparison

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

svm_model = SVC(kernel='linear')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

rf_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
print("Random Forest and KNN models trained for comparison.")
```

### Train and Evaluate the SVM Model

```python
from sklearn.metrics import accuracy_score, classification_report

svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(f"Classification Report for SVM:\n{classification_report(y_test, y_pred_svm)}")
```

### Visualize Predictions for SVM Model

```python
def plot_svm_predictions(model, X_test, y_test):
    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    y_pred = model.predict(X_test)
    
    for j, ax in enumerate(axes):
        ax.imshow(X_test[j].reshape(8, 8), cmap='gray')
        ax.set_title(f"Pred: {y_pred[j]}\nTrue: {y_test[j]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

plot_svm_predictions(svm_model, X_test, y_test)
```

## Results & Analysis

The **SVM model outperforms both Random Forest and KNN** in classifying handwritten digits with high accuracy. Below is a visualization of sample predictions made by the SVM model.

### Sample Digits from Dataset
![download](https://github.com/user-attachments/assets/e682165e-b1bd-42d4-bab1-af5967442ea7)

### SVM Model Performance Visualization
![download](https://github.com/user-attachments/assets/1233cd62-50b2-4f86-b4e4-a89bad853477)

## Why SVM?

- **Higher accuracy** compared to Random Forest and KNN.\
- **Performs well on high-dimensional data** like digit images.\
- **Works efficiently with limited data samples.**\
- **Robust to overfitting** when using a linear kernel.

## Future Improvements

- Experiment with deep learning models like **CNNs** for better performance.\
- Fine-tune hyperparameters for improved accuracy.\
- Extend the project to recognize handwritten **characters beyond digits**.\
- Deploy the model as a **web application** for real-time digit classification.

## License

This project is open-source and available under the **MIT License**.

---

**✨ Ready to classify some digits? Run the notebook and get started! 🚀**
