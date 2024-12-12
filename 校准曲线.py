import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 读取数据
df = pd.read_excel('./104.xlsx')
x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

# 数据标准化
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
x = (x - mean) / (std + 1e-10)

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.8, random_state=0)

# 模型定义和训练
adaboost_classifier = AdaBoostClassifier(n_estimators=100, random_state=42)
adaboost_classifier.fit(x_train, y_train)

forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)

# Logistic Regression 模型
logistic = LogisticRegression(max_iter=1000, random_state=0)
logistic.fit(x_train, y_train)

# XGBoost 模型
xgboost = XGBClassifier(n_estimators=100, random_state=0, use_label_encoder=False, eval_metric='logloss')
xgboost.fit(x_train, y_train)

# 模型校准 (AdaBoost和SVM)
calibrated_adaboost_isotonic = CalibratedClassifierCV(adaboost_classifier, method='isotonic', cv='prefit')
calibrated_adaboost_isotonic.fit(x_train1, y_train1)

clf = SVC(probability=True)
clf.fit(x_train, y_train)
calibrated_svm_platt = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
calibrated_svm_platt.fit(x_train1, y_train1)

# 校准 Logistic Regression 和 XGBoost
calibrated_logistic = CalibratedClassifierCV(logistic, method='isotonic', cv='prefit')
calibrated_logistic.fit(x_train1, y_train1)

calibrated_xgboost = CalibratedClassifierCV(xgboost, method='sigmoid', cv='prefit')
calibrated_xgboost.fit(x_train1, y_train1)

# 神经网络模型
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test1_scaled = scaler.transform(x_test1)
x_train_torch = torch.tensor(x_train_scaled, dtype=torch.float32)
x_test1_torch = torch.tensor(x_test1_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
y_test1_torch = torch.tensor(y_test1, dtype=torch.long)

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

input_dim = x_train_torch.shape[1]
output_dim = len(np.unique(y))

model = Net(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()

def predict_proba(model, x):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)
    return probabilities

# 计算各模型的概率预测 (使用 x_test1)
y_pred_prob_adaboost_isotonic = calibrated_adaboost_isotonic.predict_proba(x_test1)[:, 1]
y_pred_prob_rf = forest.predict_proba(x_test1)[:, 1]
y_pred_prob_nn = predict_proba(model, x_test1_torch)[:, 1].numpy()
y_pred_prob_svm_platt = calibrated_svm_platt.predict_proba(x_test1)[:, 1]
y_pred_prob_logistic = calibrated_logistic.predict_proba(x_test1)[:, 1]
y_pred_prob_xgboost = calibrated_xgboost.predict_proba(x_test1)[:, 1]

# 计算 Brier 分数
brier_adaboost_isotonic = brier_score_loss(y_test1, y_pred_prob_adaboost_isotonic)
brier_rf = brier_score_loss(y_test1, y_pred_prob_rf)
brier_nn = brier_score_loss(y_test1, y_pred_prob_nn)
brier_svm_platt = brier_score_loss(y_test1, y_pred_prob_svm_platt)
brier_logistic = brier_score_loss(y_test1, y_pred_prob_logistic)
brier_xgboost = brier_score_loss(y_test1, y_pred_prob_xgboost)

# 手动分箱函数
def manual_bin(y_true, y_prob, bins):
    bin_indices = np.digitize(y_prob, bins) - 1
    prob_true = np.array([y_true[bin_indices == i].mean() for i in range(len(bins) - 1)])
    prob_pred = np.array([y_prob[bin_indices == i].mean() for i in range(len(bins) - 1)])
    return prob_true, prob_pred

# 使用等频率分箱
def manual_bin_eqfreq(y_true, y_prob, n_bins):
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y_prob, quantiles)
    bins[-1] = 1.0
    return manual_bin(y_true, y_prob, bins)

# 定义等频率分箱的分箱数量
n_bins_adaboost_svm_nn =12
n_bins_rf = 10

# 绘制校正曲线
plt.figure(figsize=(10, 8))

# AdaBoost
prob_true, prob_pred = manual_bin_eqfreq(y_test1, y_pred_prob_adaboost_isotonic, n_bins_adaboost_svm_nn)
plt.plot(prob_pred, prob_true, marker='o', color='aqua', label=f'AdaBoost (Brier = {brier_adaboost_isotonic:.3f})')

# Random Forest
prob_true, prob_pred = manual_bin_eqfreq(y_test1, y_pred_prob_rf, n_bins_rf)
plt.plot(prob_pred, prob_true, marker='o', color='darkorange', label=f'Random Forest (Brier = {brier_rf:.3f})')

# Neural Network
prob_true, prob_pred = manual_bin_eqfreq(y_test1, y_pred_prob_nn, n_bins_adaboost_svm_nn)
plt.plot(prob_pred, prob_true, marker='o', color='cornflowerblue', label=f'Neural Network (Brier = {brier_nn:.3f})')

# SVM
prob_true, prob_pred = manual_bin_eqfreq(y_test1, y_pred_prob_svm_platt, n_bins_adaboost_svm_nn)
plt.plot(prob_pred, prob_true, marker='o', color='green', label=f'SVM (Brier = {brier_svm_platt:.3f})')

# Logistic Regression
prob_true, prob_pred = manual_bin_eqfreq(y_test1, y_pred_prob_logistic, n_bins_adaboost_svm_nn)
plt.plot(prob_pred, prob_true, marker='o', color='red', label=f'Logistic Regression (Brier = {brier_logistic:.3f})')

# XGBoost
prob_true, prob_pred = manual_bin_eqfreq(y_test1, y_pred_prob_xgboost, n_bins_adaboost_svm_nn)
plt.plot(prob_pred, prob_true, marker='o', color='blue', label=f'XGBoost (Brier = {brier_xgboost:.3f})')

# 绘制对角线
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# 设置图形属性
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curves with Brier Scores')
plt.legend(loc="lower right")
plt.show()








