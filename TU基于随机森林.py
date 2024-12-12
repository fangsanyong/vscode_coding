import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# 修正SHAP值，确保正确的格式
shap_values_with_base = shap_values[0]  # 选择第一个样本的SHAP值

# 绘制瀑布图
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_with_base, max_display=len(feature_names), show=False)  # 只选择第一个输出的解释对象
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("Waterfall_Plot.png")
plt.show()

# 绘制条形图
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values_with_base, max_display=len(feature_names), show=False)  # 只选择第一个输出的解释对象
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("Bar_Plot.png")
plt.show()

# 初始化JavaScript支持
shap.initjs()

# 绘制力图
force_plot = shap.force_plot(explainer.expected_value, shap_values.values[10], X.iloc[10], matplotlib=True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("force_Plot.png")
plt.show()
