import os
import numpy as np
import tensorflow as tf

def quantize_model(INPUT_SIZE, pb_path, output_path, calib_num, image_dir):
    # 获取本地图片路径
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    image_paths = image_paths[:calib_num]  # 仅选择校准数量的图片

    # 定义校准数据生成器
    def representative_dataset_gen():
        for i, image_path in enumerate(image_paths):
            print(f'Calibrating image {i + 1}/{len(image_paths)}: {image_path}')
            # 加载图片
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            # 调整大小
            image = tf.image.resize(image, (INPUT_SIZE, INPUT_SIZE))
            # 归一化到 [0, 1]
            image = image / 255.0
            # 扩展维度以匹配模型输入形状
            image = tf.expand_dims(image, axis=0)
            yield [image]

    input_arrays = ['inputs']
    output_arrays = ['Identity', 'Identity_1', 'Identity_2']

    # TFLite 转换器配置
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_path, input_arrays, output_arrays)
    #converter.experimental_new_quantizer = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.allow_custom_ops = False
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    converter.representative_dataset = representative_dataset_gen

    # 开始转换
    tflite_model = converter.convert()
    with open(output_path, 'wb') as w:
        w.write(tflite_model)
    print('Quantization Completed!', output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=640)
    parser.add_argument('--pb_path', default="./tflite/model_float32.pb")
    parser.add_argument('--output_path', default='./tflite/model_quantized.tflite')
    parser.add_argument('--calib_num', type=int, default=100, help='Number of images for calibration.')
    parser.add_argument('--image_dir', required=True, help='Directory containing calibration images.')
    args = parser.parse_args()
    quantize_model(args.input_size, args.pb_path, args.output_path, args.calib_num, args.image_dir)
