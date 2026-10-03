# 🍷 Wine Classification using Random Forest

A machine learning project that uses the **Wine dataset** from Scikit-learn to classify wine samples into their respective classes using a **Random Forest Classifier**.

## 📌 Project Overview

This project demonstrates a basic machine learning classification workflow:

* Loading the Wine dataset
* Creating a Pandas DataFrame
* Separating features and target
* Splitting the data into training and testing sets
* Standardizing the features
* Training a Random Forest classification model
* Making predictions
* Evaluating the model using accuracy

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

## 📊 Dataset

The project uses the built-in **Wine dataset** available through `sklearn.datasets`.

The dataset contains **13 features** used to classify wine samples into different target classes.

## 🔄 Machine Learning Workflow

### 1. Load the Dataset

```python
from sklearn.datasets import load_wine

dataset = load_wine()
```

### 2. Create DataFrame

```python
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
```

### 3. Separate Features and Target

```python
X = df.iloc[:, 0:13].values
y = df.iloc[:, 13].values
```

### 4. Train-Test Split

The dataset is divided into training and testing data using an 80/20 split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0
)
```

### 5. Feature Scaling

`StandardScaler` is used to standardize the training and testing features.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 6. Random Forest Model

A Random Forest Classifier is created with the following parameters:

```python
model = RandomForestClassifier(
    n_estimators=10,
    max_depth=10,
    max_leaf_nodes=7
)
```

The model is then trained using the training data.

### 7. Prediction

```python
y_pred = model.predict(X_test)
```

### 8. Model Evaluation

The model's accuracy is calculated using `accuracy_score`.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

## 🧪 Sample Predictions

The trained model is also used to predict the class of new wine samples:

```python
model.predict([[14.2, 1.7, 2.4, 15.6, 127, 2.8, 3.0,
                0.3, 2.0, 5.5, 1.0, 3.2, 1000]])
```

Another sample prediction:

```python
model.predict([[60, 30, 40, 50, 60, 70, 10,
                12, 23, 54, 67, 42, 15]])
```

## 📁 Project Structure

```text
wine-classification/
│
├── 2_oct_wine_practical.py
└── README.md
```

## ▶️ How to Run

### Option 1: Google Colab

1. Open the Python file in Google Colab.
2. Run the cells/code sequentially.
3. View the model accuracy and predictions.

### Option 2: Local Python Environment

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then run:

```bash
python 2_oct_wine_practical.py
```

## 📈 Model

**Algorithm:** Random Forest Classifier

Parameters used:

| Parameter            | Value |
| -------------------- | ----: |
| Number of Estimators |    10 |
| Maximum Depth        |    10 |
| Maximum Leaf Nodes   |     7 |
| Test Size            |   20% |
| Random State         |     0 |

## 🎯 Learning Objectives

This project is useful for understanding:

* Classification problems
* Dataset preprocessing
* Feature scaling
* Train-test splitting
* Random Forest classification
* Making predictions
* Model accuracy evaluation

## 👩‍💻 Author

**Trupti Priyadarshi**

---

⭐ If you found this project useful, consider giving the repository a star!
