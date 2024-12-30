from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
# 使用 iris.data 文件作为测试数据
test_file_path = 'E:/Lris/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
test_data = pd.read_csv(test_file_path, header=None, names=column_names)

# 将目标列（分类标签）映射为数值
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
test_data['target'] = test_data['target'].map(class_mapping)

# 分离特征和目标
X_test = test_data.drop('target', axis=1)  # 特征
y_test = test_data['target']  # 真实标签

# 重新创建并拟合标准化器
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)  # 注意：这会对测试数据进行新的标准化

# 加载训练好的模型
model_path = 'iris_classifier_model.joblib'
model = joblib.load(model_path)

# 使用模型进行预测
y_pred = model.predict(X_test_scaled)

# 打印预测结果
for i in range(len(X_test)):
    print(f"样本 {i + 1}: 特征值: {X_test.iloc[i].values}, Ground Truth: {y_test.iloc[i]}, Prediction: {y_pred[i]}")
