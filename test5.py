import sys
import cv2
import pandas as pd
import torch
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO

# Проверяем доступность GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

class FrameProcessingThread(QThread):
    update_frame_signal = pyqtSignal(QImage)
    send_violations_signal = pyqtSignal(list)
    frame_number = 0

    def __init__(self, frame, fps):
        super().__init__()
        self.frame = frame
        self.fps = fps

        # Загружаем модель YOLO на доступное устройство
        self.model = YOLO("best.pt").to(device)

    def run(self):
        # Оптимизация обработки изображения
        resized_frame = cv2.resize(self.frame, (640, 640))
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Запускаем инференс на доступном устройстве
        with torch.no_grad():
            results = self.model(frame_rgb, device=device)

        frame = results[0].plot()  # Визуализация детекции

        people_detected = []
        helmets_detected = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.7:  # Класс 0 - человек
                    people_detected.append(box.xyxy[0])
                elif cls == 1 and conf > 0.7:  # Класс 1 - шлем
                    helmets_detected.append(box.xyxy[0])

        violations = []
        for person in people_detected:
            person_center_x = (person[0] + person[2]) / 2
            person_center_y = person[1]

            has_helmet = False
            for helmet in helmets_detected:
                helmet_center_x = (helmet[0] + helmet[2]) / 2
                helmet_center_y = helmet[1]

                if abs(person_center_x - helmet_center_x) < 35 and abs(person_center_y - helmet_center_y) < 35:
                    has_helmet = True
                    break

            if not has_helmet:
                timestamp = FrameProcessingThread.frame_number / self.fps
                violations.append({"Время (сек)": round(timestamp, 2)})

        FrameProcessingThread.frame_number += 1

        # Преобразуем изображение для PyQt6
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.update_frame_signal.emit(q_img)

        # Отправляем результаты (нарушения) обратно в основной поток
        self.send_violations_signal.emit(violations)


class VideoProcessingThread(QThread):
    process_finished_signal = pyqtSignal(str)
    update_frame_signal = pyqtSignal(QImage)
    send_violations_signal = pyqtSignal(list)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.violations = []  # Список нарушений
        self.model = YOLO("hemletYoloV8_100epochs.pt").to(device)  # Загружаем модель на доступное устройство

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.process_finished_signal.emit("Ошибка: не удалось открыть видео")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if device == "cuda":
                # Используем CUDA для ускорения
                frame_gpu = cv2.cuda_GpuMat()
                frame_gpu.upload(frame)
                resized_frame = cv2.cuda.resize(frame_gpu, (640, 640))
                frame_rgb = cv2.cuda.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                frame_cpu = frame_rgb.download()
            else:
                # Используем CPU
                resized_frame = cv2.resize(frame, (640, 640))
                frame_cpu = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                results = self.model(frame_cpu, device=device)

            frame_cpu = results[0].plot()  # Визуализация детекции

            rgb_image = cv2.cvtColor(frame_cpu, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            q_img = QImage(rgb_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.update_frame_signal.emit(q_img)

            frame_count += 1

        cap.release()

        # Сохранение отчета
        report_df = pd.DataFrame(self.violations)
        report_path = "violations_report.csv"
        report_df.to_csv(report_path, index=False)
        self.process_finished_signal.emit(f"Отчет сохранен: {report_path}")

    def aggregate_violations(self, new_violations):
        self.violations.extend(new_violations)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Детекция людей без каски")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel("Выберите видео", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFixedSize(720, 540)

        self.load_button = QPushButton("Загрузить видео", self)
        self.load_button.clicked.connect(self.load_video)

        self.detect_button = QPushButton("Запустить детекцию", self)
        self.detect_button.clicked.connect(self.start_detection)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.detect_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.video_path = None
        self.thread = None

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Видео (*.mp4 *.avi *.mov)")
        if self.video_path:
            self.label.setText(f"Выбрано: {self.video_path}")

    def start_detection(self):
        if self.video_path:
            self.thread = VideoProcessingThread(self.video_path)
            self.thread.update_frame_signal.connect(self.update_image)
            self.thread.process_finished_signal.connect(self.show_message)
            self.thread.start()

    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)

    def show_message(self, message):
        self.label.setText(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())