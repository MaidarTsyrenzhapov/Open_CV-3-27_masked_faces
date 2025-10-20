import cv2
import numpy as np
import argparse
from typing import Tuple, List, Optional
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Загружает изображение по указанному пути.
    
    Args:
        image_path (str): Путь к изображению
        
    Returns:
        np.ndarray: Загруженное изображение в формате BGR
        
    Raises:
        FileNotFoundError: Если изображение не может быть загружено
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image from: {image_path}")
    return image


def detect_faces(image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5, 
                min_size: int = 60) -> List[Tuple[int, int, int, int]]:
    """
    Обнаруживает лица на изображении с использованием каскадов Хаара.
    
    Args:
        image (np.ndarray): Входное изображение
        scale_factor (float): Коэффициент масштабирования для пирамиды изображений
        min_neighbors (int): Минимальное количество соседей для удержания детекции
        min_size (int): Минимальный размер лица в пикселях
        
    Returns:
        List[Tuple[int, int, int, int]]: Список ограничивающих прямоугольников лиц (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Загрузка каскада для обнаружения лиц
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade classifier")
    
    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size)
    )
    
    return list(faces)


def check_mask_in_face(image: np.ndarray, face_roi: Tuple[int, int, int, int]) -> bool:
    """
    Проверяет наличие маски в области лица по цвету (красный, синий, белый/голубой).
    
    Функция анализирует нижнюю часть лица (область рта и подбородка),
    преобразует её в HSV-пространство и проверяет наличие пикселей,
    соответствующих типичным цветам медицинских масок.

    Args:
        image (np.ndarray): Исходное изображение (в формате BGR)
        face_roi (Tuple[int, int, int, int]): Область лица (x, y, w, h)

    Returns:
        bool: True — если маска обнаружена, False — если маски нет
    """
    x, y, w, h = face_roi
    
    # Определяем область нижней части лица (рот и подбородок)
    mouth_y_start = y + int(h * 0.6)
    mouth_y_end = y + int(h * 0.9)
    mouth_x_start = x + int(w * 0.2)
    mouth_x_end = x + int(w * 0.8)

    # Извлекаем область интереса
    mouth_roi = image[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
    if mouth_roi.size == 0:
        return False

    # Переводим в HSV
    hsv = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)

    # Диапазоны для типичных цветов масок
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_light = np.array([70, 0, 80])    # бело-голубые тона
    upper_light = np.array([140, 80, 255])

    # Создание бинарных масок
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    light_mask = cv2.inRange(hsv, lower_light, upper_light)

    # Объединяем все маски
    mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, blue_mask), light_mask)

    # Морфологическая очистка маски
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Вычисляем процент области, покрытой цветом маски
    mask_ratio = np.sum(mask > 0) / mask.size

    # Порог: если более 10% нижней части лица покрыто "цветом маски"
    return mask_ratio > 0.1


def process_faces_with_masks(image_path: str, output_path: str = "output.jpg") -> Tuple[int, int]:
    """
    Основная функция для обработки изображения: обнаружение лиц и проверка масок.
    
    Args:
        image_path (str): Путь к входному изображению
        output_path (str): Путь для сохранения результата
        
    Returns:
        Tuple[int, int]: Количество лиц и количество людей в масках
        
    Raises:
        Exception: При ошибках обработки
    """
    try:
        # Загрузка изображения
        image = load_image(image_path)
        output_image = image.copy()
        
        # Обнаружение лиц
        faces = detect_faces(image)
        
        masked_count = 0
        face_count = len(faces)
        
        # Обработка каждого лица
        for i, (x, y, w, h) in enumerate(faces):
            # Проверяем наличие маски
            has_mask = check_mask_in_face(image, (x, y, w, h))
            
            # Выбираем цвет рамки в зависимости от наличия маски
            color = (0, 255, 0) if has_mask else (0, 0, 255)  # Зеленый - с маской, Красный - без
            label = "Mask" if has_mask else "No Mask"
            
            # Рисуем рамку вокруг лица
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            
            # Добавляем текст с номером и статусом
            cv2.putText(output_image, f"{i+1}: {label}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if has_mask:
                masked_count += 1
        
        # Сохраняем результат
        cv2.imwrite(output_path, output_image)
        
        return face_count, masked_count
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def main():
    """
    Основная функция для запуска из командной строки.
    """
    parser = argparse.ArgumentParser(description="Detect faces and check for masks")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="output.jpg", 
                       help="Output image path (default: output.jpg)")
    parser.add_argument("--scale", type=float, default=1.1,
                       help="Scale factor for face detection (default: 1.1)")
    parser.add_argument("--neighbors", type=int, default=5,
                       help="Minimum neighbors for face detection (default: 5)")
    parser.add_argument("--minsize", type=int, default=60,
                       help="Minimum face size in pixels (default: 60)")
    
    args = parser.parse_args()
    
    try:
        # Обрабатываем изображение
        total_faces, masked_faces = process_faces_with_masks(
            args.image_path, 
            args.output
        )
        
        # Выводим результаты
        print(f"=== Face Mask Detection Results ===")
        print(f"Total faces detected: {total_faces}")
        print(f"Faces with masks: {masked_faces}")
        print(f"Faces without masks: {total_faces - masked_faces}")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())