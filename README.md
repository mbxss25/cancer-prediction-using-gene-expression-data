# 🧬 Cancer Prediction Using Gene Expression Data

This project demonstrates how to build a **machine learning pipeline** to classify cancer types based on **gene expression data**.  
It includes **data preprocessing, feature selection, model training, evaluation, and visualization**.  

## 📌 Project Description

Machine learning pipeline for cancer type prediction using gene expression data. Includes preprocessing, feature selection, Random Forest classification, performance evaluation (accuracy, precision, recall, F1), confusion matrix, and ROC-AUC visualization.

## 📂 Step 1: Importing Libraries
python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    classification_report, plot_confusion_matrix, roc_curve, auc

## 📊 Step 2: Loading and Inspecting Data
file_path = "cancer_gene_expression.csv"
data = pd.read_csv(file_path)
print(data.shape)
print(data.columns[0:5])
print(data.columns[-1])
Dataset Shape: (801, 8001) → 801 samples × 8000 gene expression features + 1 target column (Cancer_Type)
Target Classes:
BRCA – Breast Cancer
KIRC – Kidney Renal Clear Cell Carcinoma
LUAD – Lung Adenocarcinoma
PRAD – Prostate Adenocarcinoma
COAD – Colon Adenocarcinoma
🔍 Missing Values Check
datanul = data.isnull().sum()
g = [i for i in datanul if i > 0]
print(f'Columns with missing value: {len(g)}')
✅ No missing values were found.

## 🧹 Step 3: Data Preprocessing
3.1 Feature & Target Separation
X = data.iloc[:, 0:-1]  # Gene expression features
Y = data.iloc[:, -1]    # Target cancer type
X → Shape (801, 8000)
Y → Shape (801,)

3.2 Label Encoding
label_encoder = LabelEncoder()
label_encoder.fit(Y)
y_encoded = label_encoder.transform(Y)
labels = label_encoder.classes_
classes = np.unique(y_encoded)
Converts categorical cancer types into numeric labels.

3.3 Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
80% Training, 20% Testing

3.4 Normalization
min_max_scaler = MinMaxScaler()
x_train_norm = min_max_scaler.fit_transform(x_train)
x_test_norm = min_max_scaler.transform(x_test)
Scales values to [0, 1] for consistency.

## 🔎 Step 4: Feature Selection
We use Mutual Information to identify and select the most informative features.
MI = mutual_info_classif(x_train_norm, y_train)
n_features = 300
selected_score_indices = np.argsort(MI)[::-1][0:n_features]

x_train_selected = x_train_norm[:, selected_score_indices]
x_test_selected = x_test_norm[:, selected_score_indices]
✅ Only the top 300 most relevant genes are used for classification.

## 🌲 Step 5: Model Building – Random Forest Classifier
We train a Random Forest Classifier in a One-vs-Rest setting for multi-class classification.
RF = OneVsRestClassifier(RandomForestClassifier(max_features=0.2))
RF.fit(x_train_selected, y_train)
y_pred = RF.predict(x_test_selected)
pred_prob = RF.predict_proba(x_test_selected)
Random Forest: Robust ensemble model that handles high-dimensional data
One-vs-Rest: Builds one classifier per class

## 📈 Step 6: Model Evaluation
6.1 Performance Metrics
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1score = f1_score(y_test, y_pred, average='weighted')

print(classification_report(y_test, y_pred, target_names=labels))
Metric	Score
Balanced Accuracy	0.9633
Precision	0.9759
Recall	0.9752
F1-Score	0.9750
✅ Model achieves ~98% accuracy – excellent performance.

6.2 Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(pd.DataFrame(cm, index=labels, columns=labels), annot=True, cmap="Blues")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
Shows very few misclassifications — high reliability.

6.3 ROC-AUC Curves
We generate ROC Curves for all classes.
y_test_binarized = label_binarize(y_test, classes=classes)
for i in range(classes.shape[0]):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], pred_prob[:, i])
    plt.plot(fpr[i], tpr[i], linestyle='--', label=f'{labels[i]} vs Rest (AUC={roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'b--')
plt.title('Multiclass ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
✅ High AUC values across all classes confirm strong class separability.

## 🏆 Summary
Step	Technique Used	Purpose
Data Preprocessing	Label Encoding, MinMax Scaling	Prepare and normalize data
Feature Selection	Mutual Information	Choose most relevant genes
Model	Random Forest (OvR)	Multi-class classification
Evaluation	Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC	Assess performance

## Final Result:
✅ Achieved 98% classification accuracy – model effectively distinguishes cancer types.
