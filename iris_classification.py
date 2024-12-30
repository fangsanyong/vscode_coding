# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import joblib

# 1. Prepare Problem
# a)  Load libraries
# iris = load_iris()
# data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# b)  Load dataset
# 定义文件路径
file_path = 'E:/Lris/iris/bezdekIris.data'
# 手动加载数据
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']  # 定义列名
data = pd.read_csv(file_path, header=None, names=column_names)
# 将目标列（分类标签）转换为数字
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data['target'] = data['target'].map(class_mapping)
# 检查数据集
#print(data.head())


# 2. Summarize Data
# a) Descriptive statistics
print(data.describe())  # Show basic statistics of the dataset
print(data.isnull().sum())  # Check for missing values

# b) Data Visualizations
sns.pairplot(data, hue='target', height=2.5)  # Visualize pairwise relationships
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')  # Correlation heatmap
plt.show()

# 3. Prepare Data
# a) Data Cleaning
# No cleaning needed as Iris dataset is clean

# b) Feature Selection
# All features are used for this simple example

# c) Data Transforms
scaler = StandardScaler()
X = data.drop('target', axis=1)
y = data['target']
X_scaled = scaler.fit_transform(X)  # Scale features

# 4. Evaluate Algorithms
# a) Split dataset for train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# b) Test options and evaluation metric
# We'll use accuracy, precision, recall, F1-score for evaluation

# c) Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))

# d) Compare algorithms on train set via cross validation
for name, model in models:
    kfold = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: {kfold.mean():.3f} ({kfold.std():.3f})")

# 5. Improve Accuracy
# a) Algorithm Tuning
param_grid = {
    'n_estimators': [10, 50, 100, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)

# b) Ensembles
voting_clf = VotingClassifier(estimators=[('lr', LogisticRegression()), 
                                          ('rf', RandomForestClassifier(n_estimators=100)), 
                                          ('svc', SVC(probability=True))], voting='soft')
voting_clf.fit(X_train, y_train)

# 6. Finalize Model
# a) Create standalone model on entire training dataset
final_model = voting_clf

# b) Predictions on test dataset
y_pred = final_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# c) Save model for later use
joblib.dump(final_model, 'iris_classifier_model.joblib')