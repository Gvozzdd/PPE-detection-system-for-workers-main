import sys
import cv2
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QTextBrowser, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QImage, QPixmap, QIcon
from ultralytics import YOLO
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class VideoProcessingThread(QThread):
    update_frame_signal = pyqtSignal(QImage)
    process_finished_signal = pyqtSignal(list, int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.model = YOLO("hemletYoloV8_100epochs.pt")
        self.tracked_people = {}  # ID -> [количество кадров без каски]
        self.violations_set = set()  # Для уникальных нарушений
        self.conf_threshold = 0.6  # Увеличенный порог уверенности
        self.required_frames = 3  # Количество кадров для уверенного нарушения

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть видео")
            self.process_finished_signal.emit([], 0)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        violations = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(frame, persist=True, conf=self.conf_threshold, iou=0.5)
            current_frame_ids = set()

            if results[0].boxes.id is not None:
                for box, obj_id in zip(results[0].boxes, results[0].boxes.id):
                    cls = int(box.cls[0])  # Класс объекта (0 - без каски, 1 - в каске)
                    obj_id = int(obj_id)  # ID объекта
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    current_frame_ids.add(obj_id)

                    # Игнорируем слишком маленькие объекты
                    if (x2 - x1) * (y2 - y1) < 500:
                        continue

                    if cls == 0:  # Без каски
                        if obj_id not in self.tracked_people:
                            self.tracked_people[obj_id] = 1
                        else:
                            self.tracked_people[obj_id] += 1

                        # Запоминаем нарушение, если оно длится несколько кадров подряд
                        if self.tracked_people[obj_id] >= self.required_frames:
                            if obj_id not in self.violations_set:
                                timestamp = round(frame_count / fps, 2)
                                violations.append(timestamp)
                                self.violations_set.add(obj_id)
                        color = (0, 0, 255)  # Красный - нарушение
                    else:
                        self.tracked_people[obj_id] = 0  # Сброс счетчика, если человек в каске
                        color = (0, 255, 0)  # Зеленый - в каске

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            frame_count += 1
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            q_img = QImage(rgb_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.update_frame_signal.emit(q_img)

        cap.release()
        print(f"Итоговое количество нарушителей: {len(self.violations_set)}")
        pd.DataFrame({"Время (сек)": violations}).to_csv("violations_report.csv", index=False)
        self.process_finished_signal.emit(violations, len(self.violations_set))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Детекция людей без каски")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("icon.png"))  # Устанавливаем иконку окна

        # Устанавливаем стиль для окна
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f2f2f2;
            }
            QLabel {
                font-size: 18px;
                color: #333;
                font-weight: bold;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 15px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTextBrowser {
                border: 1px solid #ccc;
                background-color: #fff;
                padding: 10px;
            }
            QWidget {
                font-family: Arial, sans-serif;
            }
        """)

        self.label = QLabel("Выберите видео", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFixedSize(720, 400)

        self.text_report = QTextBrowser(self)
        self.text_report.setFixedSize(720, 100)
        self.text_report.anchorClicked.connect(self.handle_link_click)

        self.load_button = QPushButton("Загрузить видео", self)
        self.load_button.clicked.connect(self.load_video)

        self.detect_button = QPushButton("Запустить детекцию", self)
        self.detect_button.clicked.connect(self.start_detection)

        # Расположение кнопок и текстового поля
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.detect_button)
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.text_report)
        layout.addLayout(buttons_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.video_path = None
        self.thread = None
        self.violations = []
        self.total_violations = 0

        # Обновляем интерфейс для полноэкранного режима
        self.setMinimumSize(800, 600)

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Видео (*.mp4 *.avi *.mov)")
        if self.video_path:
            self.label.setText(f"Выбрано: {self.video_path}")

    def start_detection(self):
        if self.video_path:
            self.thread = VideoProcessingThread(self.video_path)
            self.thread.update_frame_signal.connect(self.update_image)
            self.thread.process_finished_signal.connect(self.show_report)
            self.thread.start()

    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)

    def show_report(self, violations, total_violations):
        self.violations = violations
        self.total_violations = total_violations
        if violations:
            report_message = f"<h3>Обнаружено {total_violations} человек без касок.</h3><br><b>Тайминги:</b><br>"
            report_message += " ".join(f'<a href="{v}">{v} сек</a>' for v in violations)
        else:
            report_message = "<h3>Нарушения не обнаружены.</h3>"
        self.text_report.setHtml(report_message)

    def handle_link_click(self, url):
        timestamp = float(url.toString())
        self.jump_to_time(timestamp)
        self.show_report(self.violations, self.total_violations)
    
    def jump_to_time(self, timestamp):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            q_img = QImage(rgb_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.update_image(q_img)
        cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()  # Открываем окно на весь экран
    sys.exit(app.exec())
