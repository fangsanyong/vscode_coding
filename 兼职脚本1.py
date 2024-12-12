import os
import cv2
import numpy as np

# 定义函数来反解坐标并绘制框，增加类别显示
def draw_bounding_boxes(image_path, label_path, class_names):
    # 读取图片
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 打开对应的txt文件，读取标注信息
    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # 解析每行数据
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        # 反解坐标到图像坐标系
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        w = int(w * width)
        h = int(h * height)

        # 计算左上角和右下角坐标
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # 随机选择一个颜色来表示不同的类别
        color = class_colors[class_id]

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 显示类别名称
        label = class_names[class_id]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label, (x1, y1 - 10), font, 0.5, color, 2, cv2.LINE_AA)

    # 保存带框的图像
    output_path = image_path.replace('.jpg', '_boxed.jpg')
    cv2.imwrite(output_path, image)
    print(f"Saved boxed image to {output_path}")

# 设置图像和标注文件夹路径
image_folder = 'C:/Users/ThinkBook/Desktop/show'  # 图像文件夹路径
label_folder = 'C:/Users/ThinkBook/Desktop/show'
# 获取所有图像文件

# 目标类别名称列表 (根据YOLO标注的类别来修改)
class_names = ["class_0", "class_1", "class_2"]  # 示例类别名，实际应根据自己的数据集修改

# 颜色字典：为每个类别分配一个颜色
class_colors = {
    0: (0, 255, 0),    # 绿色
    1: (255, 0, 0),    # 红色
    2: (0, 0, 255),    # 蓝色
    # 可以继续为其他类别添加颜色
}

# 获取所有图像文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 对每张图片进行标注反解并保存结果
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    label_file = image_file.replace('.jpg', '.txt')
    label_path = os.path.join(label_folder, label_file)

    # 确保对应的标注文件存在
    if os.path.exists(label_path):
        draw_bounding_boxes(image_path, label_path, class_names)
    else:
        print(f"Label file {label_file} not found, skipping.")
