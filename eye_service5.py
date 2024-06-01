from datetime import datetime

import cv2
import dlib
import os
import threading
from eye_model import predict_eye_state, load_eye_model  # Импорт функции предсказания состояния глаза
from queue import Queue

# Инициализация детектора лиц и предиктора ключевых точек
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../dlib/shape_predictor_68_face_landmarks.dat")
save_directory = 'image'
output_folder = "eye_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(0)
frame_count = 0

# Загрузка предварительно обученной модели
model_path = '../model/best_modelv4.h5'
eye_model = load_eye_model(model_path)

# Очередь для хранения изображений глаз
eye_image_queue = Queue()

# Параметр для определения частоты обрезания глаз
eye_crop_interval = 15  # По умолчанию раз в 30 кадров

# Переменная для хранения последнего состояния глаза
last_eye_state = None

# Переменная для хранения времени, в течение которого глаза были закрытыми
closed_eyes_time = 0

# Порог времени, после которого выводится текст
closed_eyes_threshold = 0.15  # 0.1 = 1 секунде

# Функция для обработки изображений глаз в другом потоке
def eye_processing_thread():
    global last_eye_state, closed_eyes_time
    while True:
        eye_image = eye_image_queue.get()
        if eye_image is None:
            break
        eye_state = predict_eye_state(eye_image, eye_model)  # Предсказание состояния глаза
        last_eye_state = eye_state  # Сохраняем последнее состояние глаза
        print("Eye state:", eye_state)

        # Проверка, закрыты ли глаза
        if eye_state > 0.5:
            closed_eyes_time += 1 / eye_crop_interval  # Увеличиваем время закрытых глаз на 1/eye_crop секунды (примерно равно интервалу между кадрами)
        if eye_state < 0.5:
            closed_eyes_time = 0
# Создание и запуск потока для обработки изображений глаз
eye_thread = threading.Thread(target=eye_processing_thread)
eye_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        if frame_count % eye_crop_interval == 0:  # Вырезаем глаз каждые eye_crop_interval кадров
            # Выбираем индексы ключевых точек для левого глаза (обычно это точки с 36 по 41)
            left_eye_indices = list(range(36, 42))
            for n in left_eye_indices:
                x_eye, y_eye = landmarks.part(n).x, landmarks.part(n).y
                eye_size = 64
                x_eye -= eye_size // 2
                y_eye -= eye_size // 2
                eye_image = gray[y_eye:y_eye + eye_size, x_eye:x_eye + eye_size]
                # Помещаем изображение глаза в очередь для последующей обработки
                eye_image_queue.put(eye_image)
                #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                #filename = f"{save_directory}/eye_{timestamp}.png"
                #cv2.imwrite(filename, eye_image)


    # Отображение лиц на кадре
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Вывод последнего состояния глаза на кадре
    if last_eye_state is not None:
        cv2.putText(frame, "Eye state: " + ("Open" if last_eye_state < 0.5 else "Close"), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Проверка, если глаза закрыты более closed_eyes_threshold секунд
    if last_eye_state is not None and last_eye_state > 0.5:
        if closed_eyes_time >= closed_eyes_threshold:
            cv2.putText(frame, "Take care!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
    cv2.imshow('Face and Eye Detection', frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Завершаем поток обработки изображений глаз
eye_image_queue.put(None)
eye_thread.join()

cap.release()
cv2.destroyAllWindows()

