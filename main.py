from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
from ultralytics import YOLO
import cv2


class VideoProcessingThread(QThread):
    frame_processed = pyqtSignal(QImage)
    processing_finished = pyqtSignal()

    def __init__(self, model, video_path, output_dir="results"):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.output_dir = output_dir
        self.stop_flag = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(self.output_dir, "output_video.mp4")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            # Применяем модель YOLO для обработки
            results = self.model.track(frame, iou=0.6, conf=0.5, persist=True, imgsz=608, verbose=False)
            processed_frame = results[0].plot()  # Отрисовка результатов на кадре

            # Сохраняем кадр в выходное видео
            out.write(processed_frame)

            # Конвертируем кадр для отображения в PyQt
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)

            # Передаём кадр в основной поток
            self.frame_processed.emit(qt_image)

            cv2.waitKey(int(1000 / fps))  # Задержка между кадрами

        cap.release()
        out.release()
        self.processing_finished.emit()


class PoseAnalyser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Analyser")
        screen = QApplication.primaryScreen().size()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))
        self.setStyleSheet("background-color: white;")

        self.font = QFont("PIXY", 18)
        self.font.setBold(True)
        self.font.setStyleStrategy(QFont.PreferAntialias)

        font_path = os.path.join(os.path.dirname(__file__), "PIXY.ttf")
        if os.path.exists(font_path):
            QFontDatabase.addApplicationFont(font_path)

        self.video_path = None
        self.model = YOLO('yolov8m-pose.pt')
        self.video_thread = None
        self.create_ui()

    def create_ui(self):
        self.background = QLabel(self)
        self.background.setPixmap(QPixmap("1.jpg"))
        self.background.setScaledContents(True)
        self.background.setGeometry(0, 0, self.width(), self.height())

        self.select_button = QPushButton("Выберите файл", self)
        self.select_button.setFont(self.font)
        self.select_button.setStyleSheet("background-color: #3a3837; color: white; border-radius: 10px;")
        self.select_button.clicked.connect(self.open_file_dialog)
        self.update_button_position(self.select_button)

        self.video_display = QLabel(self)
        self.video_display.setGeometry(0, 0, self.width(), self.height())
        self.video_display.setStyleSheet("background-color: black;")
        self.video_display.hide()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите видеофайл", "", "Video Files (*.mp4 *.avi)",
                                                   options=options)
        if file_path:
            self.video_path = file_path
            self.start_processing()

    def start_processing(self):
        self.select_button.hide()
        self.background.hide()
        self.video_display.show()

        # Запускаем поток обработки видео
        self.video_thread = VideoProcessingThread(self.model, self.video_path)
        self.video_thread.frame_processed.connect(self.update_video_frame)
        self.video_thread.processing_finished.connect(self.processing_complete)
        self.video_thread.start()

    def update_video_frame(self, frame):
        # Обновляем QLabel с текущим кадром
        self.video_display.setPixmap(QPixmap.fromImage(frame))

    def processing_complete(self):
        # Завершаем обработку
        self.video_display.hide()
        self.background.setPixmap(QPixmap("3.jpg"))
        self.background.show()

    def update_button_position(self, button):
        button_width, button_height = 200, 100
        button.setGeometry(
            (self.width() - button_width) // 2,
            (self.height() - button_height) // 2,
            button_width,
            button_height,
        )


if __name__ == "__main__":
    app = QApplication([])
    window = PoseAnalyser()
    window.show()
    app.exec_()
