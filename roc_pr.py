import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plot
from itertools import cycle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']

# Load data
df = pd.read_excel('./11065.xlsx')
x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

# Data standardization
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
x = (x - mean) / (std + 1e-12)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.81, random_state=42)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=0)

# Model 1: AdaBoost
adaboost_classifier = AdaBoostClassifier(n_estimators=100, random_state=42)
adaboost_classifier.fit(x_train, y_train)
adaboost_predictions = adaboost_classifier.predict(x_test)

# Model 2: RandomForest
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)
randomforest_predictions = forest.predict(x_test)

# Model 3: Neural Network
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = x_train.shape[1]
output_dim = len(np.unique(y))
model = Net(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Neural Network
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

def predict_proba(model, x):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)
    return probabilities

# Model 4: SVM
from sklearn.svm import SVC
clf = SVC(probability=True)
clf.fit(x_train, y_train)

# Model 5: Logistic Regression
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(x_train, y_train)

# Model 6: XGBoost
xgboost_classifier = XGBClassifier(n_estimators=100, random_state=42)
xgboost_classifier.fit(x_train, y_train)

# Classifiers for comparison
classifiers = {
    "AdaBoost": adaboost_classifier,
    "Random Forest": forest,
    "Neural Network": model,
    "SVM": clf,
    "Logistic Regression": logistic_regression,
    "XGBoost": xgboost_classifier
}

# Plot ROC curves
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'blue'])
plot.figure(figsize=(10, 8))

for (name, clf), color in zip(classifiers.items(), colors):
    if name == "Neural Network":
        y_pred_prob = predict_proba(clf, torch.tensor(x_test2, dtype=torch.float32))[:, 1].numpy()
    else:
        y_pred_prob = clf.predict_proba(x_test2)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test2, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plot.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plot.xlim([0.0, 1.0])
plot.ylim([0.0, 1.05])
plot.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plot.title('Receiver Operating Characteristic (ROC) Curves')
plot.legend(loc="lower right")
plot.show()

# Plot PR curves
plot.figure(figsize=(10, 8))
for (name, clf), color in zip(classifiers.items(), colors):
    if name == "Neural Network":
        y_pred_prob = predict_proba(clf, torch.tensor(x_test2, dtype=torch.float32))[:, 1].numpy()
    else:
        y_pred_prob = clf.predict_proba(x_test2)[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_test2, y_pred_prob)
    avg_precision = average_precision_score(y_test2, y_pred_prob)
    plot.plot(recall, precision, color=color, lw=2, label=f'{name} (AP = {avg_precision:.2f})')

plot.xlabel('Recall')
plot.ylabel('Precision')
plot.title('Precision-Recall (PR) Curves')
plot.legend(loc="lower right")
plot.show()




