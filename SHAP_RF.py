import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
#from matplotlib import font_manager

import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np
#from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 (SimHei)
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 加载数据
data = pd.read_excel('./iris_dataset.xlsx')

# 定义特征和目标变量
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
#model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 使用SHAP解释模型
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)


# 绘制 SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_train, show=False, max_display=X_train.shape[1])
plt.title("特征重要性（SHAP）值 -点图")
plt.show()

# 计算特征重要性
shap_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': np.mean(np.abs(shap_values), axis=0)
}).sort_values(by='importance', ascending=False)

# 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
plt.bar(shap_importance['feature'], shap_importance['importance'], color='steelblue')
plt.xlabel("特征")
plt.ylabel("SHAP 重要性")
plt.title("平均特征重要性")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 输出特征重要性
print(shap_importance)

# 解释单个预测
def visualize_single_prediction(index):
    plt.figure()
    shap.force_plot(
        explainer.expected_value, 
        shap_values[index], 
        X_train.iloc[index], 
        matplotlib=True, 
        show=False
    )
    plt.title(f'单个预测的SHAP解释 (样本 {index})')
    plt.show()

# 保存单个预测解释
shap.initjs()
def save_single_prediction_html(index, filename):
    force_plot = shap.force_plot(
        explainer.expected_value, 
        shap_values[index], 
        X_train.iloc[index]
    )
    shap.save_html(filename, force_plot)
    print(f"SHAP force plot has been saved to {filename}")

# 示例用法
visualize_single_prediction(0)
save_single_prediction_html(0, "shap_force_plot.html")
