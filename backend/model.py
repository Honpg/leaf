import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class SEBlock(layers.Layer):
    def __init__(self, reduction=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        filters = input_shape[-1]
        self.dense1 = layers.Dense(filters // self.reduction, activation='relu')
        self.dense2 = layers.Dense(filters, activation='sigmoid')

    def call(self, inputs):
        se = layers.GlobalAveragePooling2D()(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = tf.reshape(se, [-1, 1, 1, inputs.shape[-1]])
        return layers.Multiply()([inputs, se])
# Đăng ký lớp SEBlock vào Keras
get_custom_objects().update({'SEBlock': SEBlock})

def load_model(model_path):
    """Load mô hình từ file"""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_bytes):
    """Tiền xử lý ảnh cho mô hình"""
    img = image.load_img(image_bytes, target_size=(256, 256))  # Resize ảnh về (256, 256)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def get_prediction(model, img_array):
    """Dự đoán lớp bệnh và xác suất"""
    predictions = model.predict(img_array)
    
    class_labels = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    top_3_predictions = sorted(zip(class_labels, predictions[0]), key=lambda x: x[1], reverse=True)[:3]

    return predicted_label, confidence, top_3_predictions


