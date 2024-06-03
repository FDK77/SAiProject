import os
import cv2
import dlib
import threading
import pygame
from queue import Queue
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
import time  # Import the time module

# Import function to predict eye state
from eye_model import predict_eye_state, load_eye_model

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")
output_folder = "eye_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load pre-trained eye state model
model_path = 'model/best_modelv4.h5'
eye_model = load_eye_model(model_path)

# Queue for storing eye images
eye_image_queue = Queue()

# Parameters for eye cropping frequency
eye_crop_interval = 15  # Adjust as needed

# Variables for tracking eye state
last_eye_state = None
closed_eyes_threshold = 0.3  # 0.30 seconds

# Variables for tracking face state
last_face_detected = True
face_detection_threshold = 1  # 0.30 seconds
face_detection_start_time = None

import collections

class EyeDetectionApp(App):
    def build(self):
        self.frame_count = 0
        self.capture = cv2.VideoCapture(0)
        self.img = Image()
        self.eye_state_label = Label(text="Состояние глаз: Неизвестно", font_size='18sp', size_hint_y=0.1)
        self.face_detection_label = Label(text="Обнаружение лица: Не обнаружено", font_size='18sp', size_hint_y=0.1)
        self.alert_label = Label(text="", font_size='18sp', color=(1, 0, 0, 1), size_hint_y=0.1)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img)

        info_layout = BoxLayout(orientation='horizontal', size_hint_y=0.3)
        info_layout.add_widget(self.eye_state_label)
        info_layout.add_widget(self.face_detection_label)

        layout.add_widget(info_layout)
        layout.add_widget(self.alert_label)

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update at 30 FPS

        # Start eye processing thread
        self.eye_thread = threading.Thread(target=self.eye_processing_thread)
        self.eye_thread.start()
        pygame.mixer.init()
        self.siren_sound = pygame.mixer.Sound("sound/siren.wav")

        # Initialize smoothing window
        self.smoothing_window = collections.deque(maxlen=eye_crop_interval)
        self.closed_eyes_start_time = None  # Initialize closed_eyes_start_time

        return layout
    def update(self, dt):
        global last_face_detected, face_detection_start_time

        ret, frame = self.capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        current_time = time.time()

        if not faces:
            if last_face_detected:
                face_detection_start_time = current_time
                last_face_detected = False
            else:
                elapsed_time = current_time - face_detection_start_time
                if elapsed_time >= face_detection_threshold:
                    self.face_detection_label.text = "Обнаружение лица: Не обнаружено"
                    if not pygame.mixer.Channel(1).get_busy():
                        pygame.mixer.Channel(1).play(self.siren_sound)
        else:
            last_face_detected = True
            face_detection_start_time = None
            self.face_detection_label.text = "Обнаружение лица: Обнаружено"
            self.alert_label.text = ""
            pygame.mixer.Channel(1).stop()

            for face in faces:
                landmarks = predictor(gray, face)
                if self.frame_count % eye_crop_interval == 0:
                    left_eye_indices = list(range(36, 42))
                    for n in left_eye_indices:
                        x_eye, y_eye = landmarks.part(n).x, landmarks.part(n).y
                        eye_size = 64
                        x_eye -= eye_size // 2
                        y_eye -= eye_size // 2
                        eye_image = gray[y_eye:y_eye + eye_size, x_eye:x_eye + eye_size]
                        eye_image_queue.put(eye_image)

            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if last_eye_state is not None:
            self.smoothing_window.append(last_eye_state)
            smoothed_state = sum(self.smoothing_window) / len(self.smoothing_window)

            if smoothed_state > 0.5:
                if self.closed_eyes_start_time is None:
                    self.closed_eyes_start_time = current_time
                elapsed_time = current_time - self.closed_eyes_start_time
                if elapsed_time >= closed_eyes_threshold:
                    self.eye_state_label.text = "Состояние глаз: Закрыты"
                    self.alert_label.text = "Внимание! Следите за дорогой!"
                    if not pygame.mixer.Channel(0).get_busy():
                        pygame.mixer.Channel(0).play(self.siren_sound)
            else:
                self.closed_eyes_start_time = None
                self.eye_state_label.text = "Состояние глаз: Открыты"
                self.alert_label.text = ""
                pygame.mixer.Channel(0).stop()

        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture

        self.frame_count += 1

    def eye_processing_thread(self):
        global last_eye_state
        while True:
            eye_image = eye_image_queue.get()
            if eye_image is None:
                break
            eye_state = predict_eye_state(eye_image, eye_model)
            last_eye_state = eye_state
            print("Eye state:", eye_state)

    def on_stop(self):
        eye_image_queue.put(None)
        self.eye_thread.join()
        self.capture.release()

if __name__ == '__main__':
    EyeDetectionApp().run()
