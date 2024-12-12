import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 设置绘图参数
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']

# 加载数据
data = pd.read_excel('./11065.xlsx')

# 定义特征和目标变量
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# 训练模型 - 使用随机森林回归
model = RandomForestRegressor()
model.fit(X, y)

# 计算SHAP值
explainer = shap.Explainer(model, X)
shap_values = explainer(X, check_additivity=False)  # 禁用加性检查

# 获取特征名称列表
feature_names = X.columns

# 初始化JavaScript支持
shap.initjs()


import matplotlib.pyplot as plt
import shap

# 假设 shap_values 和 X 已定义
chosen_feature = feature_names[13]  # 选择第13个特征

# 创建图形
plt.figure(figsize=(10, 6))
shap.dependence_plot(chosen_feature, shap_values.values, X)

# 获取当前轴并添加 SHAP=0 的虚线
ax = plt.gca()
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

# 设置坐标轴字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

plt.show()
