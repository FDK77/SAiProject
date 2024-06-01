import cv2
import numpy as np
import tensorflow as tf

def load_eye_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_eye_state(eye_image, model):
    # Предобработка изображения глаза
    eye_image = cv2.resize(eye_image, (64, 64))  # Изменение размера до 64x64 (как в обучающем наборе)
    eye_image = eye_image / 255.0  # Нормализация значений пикселей

    # Добавление измерения пакета и канала
    eye_image = np.expand_dims(eye_image, axis=0)
    eye_image = np.expand_dims(eye_image, axis=-1)

    # Предсказание состояния глаза
    prediction = model.predict(eye_image)
    return prediction[0][0]  # Возвращаем вероятность закрытого глаза (от 0 до 1)
