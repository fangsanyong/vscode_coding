import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 (SimHei)
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 加载数据
data = pd.read_excel('./iris_dataset.xlsx')

# 定义特征和目标变量
X = data.iloc[:, 0:4]
y = data.iloc[:, 4]

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 定义一个模型预测函数
def model_predict(X):
    return model.predict_proba(X)  # 使用 predict_proba 以获得每个类别的概率

# 使用SHAP解释模型
explainer = shap.KernelExplainer(model_predict, X_train)
shap_values = explainer.shap_values(X_test, nsamples=100)

# 绘制 SHAP summary plot
for i, class_name in enumerate(model.classes_):
    print(f"绘制类别 {class_name} 的 SHAP summary plot...")
    plt.figure(figsize=(16, 12))  # 增大图形尺寸
    shap.summary_plot(shap_values[:,:,i], X_test, show=False,feature_names=data.columns.tolist()[1:], max_display=len(data.columns.tolist()[1:]))
    plt.title(f"类别 {class_name} 的 SHAP 特征重要性 - 随机森林模型 (Iris 数据集)")
    plt.tight_layout()  # 调整子图布局
    plt.savefig(f'shap_summary_plot_class_{class_name}.png')  # 保存图形
    plt.show()

