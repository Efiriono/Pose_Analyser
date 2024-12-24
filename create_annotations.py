import os
import json
import cv2

training_data_path = "training_data.json"
dataset_dir = "dataset"
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")
video_path = "test_video.mp4" 

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

try:
    with open(training_data_path, "r") as f:
        training_data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Файл {training_data_path} не найден.")
except json.JSONDecodeError:
    raise ValueError(f"Ошибка чтения JSON в файле {training_data_path}.")

# Открытие видео
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise FileNotFoundError(f"Не удалось открыть видеофайл {video_path}.")

# Генерация изображений и аннотаций
for entry in training_data:
    frame_number = entry.get("frame")
    person_id = entry.get("person_id")
    keypoints = entry.get("keypoints")

    if frame_number is None or person_id is None or not isinstance(keypoints, dict):
        print(f"Пропуск записи: {entry}")
        continue

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()

    if not ret or frame is None:
        print(f"Не удалось прочитать кадр {frame_number}. Пропуск.")
        continue

    image_name = f"frame_{frame_number}_person_{person_id}.jpg"
    image_path = os.path.join(images_dir, image_name)
    cv2.imwrite(image_path, frame)

    label_name = f"frame_{frame_number}_person_{person_id}.txt"
    label_path = os.path.join(labels_dir, label_name)
    with open(label_path, "w") as label_file:
        keypoints_line = "0 "  
        for point_name, coords in keypoints.items():
            if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                print(f"Некорректные координаты для {point_name}: {coords}. Пропуск.")
                continue
            x, y = coords
            keypoints_line += f"{x / frame.shape[1]} {y / frame.shape[0]} " 
        label_file.write(keypoints_line.strip() + "\n")

video.release()
print("Данные подготовлены для тренировки.")
