# CodeAlpha ML Model Training: Iris Dataset

## 📋 Project Overview

This project demonstrates a comprehensive machine learning workflow using the famous Iris dataset. It includes data exploration, visualization, preprocessing, model training, and evaluation using a Support Vector Machine (SVM) classifier.

## 🌸 Dataset Information

The **Iris dataset** is a classic dataset in machine learning containing 150 samples of iris flowers with the following characteristics:

- **Total Samples**: 150
- **Features**: 4 numerical measurements
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Classes**: 3 iris species
  - Iris-setosa (50 samples)
  - Iris-versicolor (50 samples)
  - Iris-virginica (50 samples)
- **Data Quality**: No missing values

## 📊 Project Structure

```
iris_dataset_ML.ipynb
├── Data Loading & Exploration
│   ├── Load dataset from CSV
│   ├── Display basic statistics
│   ├── Check data types and missing values
│   └── Analyze feature distributions
│
├── Data Visualization
│   ├── Scatter plots (feature relationships)
│   ├── Histogram plots (data distributions)
│   └── Box plots (outliers and spread)
│
├── Data Preparation
│   ├── Separate features (X) and target (y)
│   ├── Train-test split (80-20)
│   └── Feature scaling (StandardScaler)
│
├── Model Training
│   ├── SVM with RBF kernel
│   └── Cross-validation (5-fold)
│
└── Model Evaluation
    ├── Performance metrics
    ├── Confusion matrix
    ├── Classification report
    └── Single flower prediction
```

## 🔍 Visualizations Included

### 1. Scatter Plots
- Sepal Length vs Sepal Width
- Sepal Length vs Petal Length
- Petal Length vs Petal Width

### 2. Histogram Plots
- Distribution of all 4 numerical features
- Shows frequency distribution for each measurement

### 3. Box Plots
- Identifies outliers and data spread
- Shows median, quartiles, and range for each feature

## 🤖 Model Details

### Algorithm: Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF)
- **Purpose**: Multi-class classification
- **Reason for Choice**: Excellent performance on small to medium-sized datasets like Iris

### Data Splitting
- **Training Set**: 120 samples (80%)
- **Test Set**: 30 samples (20%)
- **Stratified Split**: Ensures balanced class distribution

### Feature Scaling
- **Scaler**: StandardScaler (z-score normalization)
- **Purpose**: Normalize features to similar scales for better SVM performance

## 📈 Model Performance

### Overall Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | 96.67% |
| **Precision** | 96.97% |
| **Recall** | 96.67% |
| **F1-Score** | 96.66% |

### Per-Class Performance
```
              Precision  Recall  F1-Score  Support
Iris-setosa      1.00     1.00     1.00       10
Iris-versicolor  1.00     0.90     0.95       10
Iris-virginica   0.91     1.00     0.95       10
```

### Confusion Matrix
```
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```

### Cross-Validation Results (5-Fold)
- Individual Fold Scores: [0.917, 1.000, 0.958, 0.958, 1.000]
- Mean CV Accuracy: 96.67%
- Standard Deviation: 0.0312

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook
```bash
jupyter notebook iris_dataset_ML.ipynb
```

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | Latest | Data manipulation and analysis |
| numpy | Latest | Numerical computations |
| matplotlib | Latest | Plotting and visualization |
| seaborn | Latest | Statistical data visualization |
| scikit-learn | Latest | Machine learning algorithms and metrics |

## 💻 Usage

### Load and Explore Data
```python
df = pd.read_csv('Iris.csv')
print(df.head())
print(df.describe())
```

### Train the Model
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='rbf', random_state=42)
model.fit(X_train_scaled, y_train)
```

### Make Predictions
```python
# Predict on test data
y_pred = model.predict(X_test_scaled)

# Predict a new flower
new_flower = [[5.8, 3.2, 4.5, 1.2]]
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
print(f"Predicted species: {prediction[0]}")
```

### Evaluate Performance
```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

## 🎯 Key Findings

1. **High Model Accuracy**: The SVM classifier achieved 96.67% accuracy on the test set
2. **Balanced Performance**: All three iris species are classified with similar accuracy
3. **Excellent Generalization**: Cross-validation shows consistent performance (96.67% ± 3.12%)
4. **Minor Misclassification**: Only 1 out of 30 test samples was misclassified
   - Iris-versicolor misclassified as Iris-virginica

## 📝 Code Sections

1. **Data Loading** - Loads Iris.csv and displays initial data
2. **Exploratory Data Analysis (EDA)** - Statistical summary and data types
3. **Data Visualization** - Creates scatter plots, histograms, and box plots
4. **Data Preparation** - Feature extraction, splitting, and scaling
5. **Model Training** - SVM model creation and training
6. **Model Evaluation** - Comprehensive performance metrics and cross-validation
7. **Prediction** - Single flower prediction example

## 🔧 Model Hyperparameters

```python
model = SVC(
    kernel='rbf',        # Radial Basis Function kernel
    random_state=42      # For reproducibility
)
```

## 📚 Learning Outcomes

This project demonstrates:
- ✅ Data loading and exploration with pandas
- ✅ Data visualization using matplotlib and seaborn
- ✅ Feature scaling for machine learning
- ✅ Train-test data splitting with stratification
- ✅ SVM classification model training
- ✅ Model evaluation and performance metrics
- ✅ Cross-validation for robustness assessment
- ✅ Making predictions on new data

## 📂 File Requirements

To run this project, you need:
- `iris_dataset_ML.ipynb` - Main Jupyter notebook
- `Iris.csv` - Dataset file (required in the same directory)

## 🤝 Contributing

Feel free to fork this project and make improvements such as:
- Testing different algorithms (Random Forest, KNN, Logistic Regression)
- Hyperparameter tuning
- Feature engineering
- ROC curve analysis
- Additional visualizations

## 📄 License

This project is provided as-is for educational purposes.

## 📧 Contact

For questions or suggestions, please reach out to the project owner.

---

**Created as part of CodeAlpha ML Model Training Program**