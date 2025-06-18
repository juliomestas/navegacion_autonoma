from controller import Display, Robot, Camera, Keyboard
from vehicle import Car, Driver
import numpy as np
import cv2
import os
import csv
import time

# Ajustes
CAPTURE_INTERVAL = 0.2  # ~5 imágenes por segundo
IMAGES_FOLDER = "captured_images"
CSV_FILE = "angles.csv"

# Crear carpeta si no existe
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

# Crear archivo CSV y encabezado si no existe
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "steering_angle"])

def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    return image[:, :, :3]

def greyscale_cv2(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_trapezoid_mask(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    vertices = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.4), int(height * 0.5)),
        (int(width * 0.6), int(height * 0.5)),
        (int(width * 0.9), height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, (255, 255, 255))
    return cv2.bitwise_and(image, mask)

def detect_lane_angle_trapezoid(image, last_angle):
    masked = apply_trapezoid_mask(image)
    gray = greyscale_cv2(masked)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=20)
    angles = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = -np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)

    if not angles:
        return last_angle, lines

    angle = np.mean(angles)
    angle = max(-0.5, min(0.5, angle))
    return angle, lines

def display_image_with_lines(display, image, lines):
    image_with_lines = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    rgba_img = np.concatenate(
        (image_with_lines, np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)), axis=2
    )
    ref = display.imageNew(rgba_img.tobytes(), Display.RGBA, rgba_img.shape[1], rgba_img.shape[0])
    display.imagePaste(ref, 0, 0, False)

def save_image_and_log(image, angle, count):
    filename = f"img_{count:05}.png"
    filepath = os.path.join(IMAGES_FOLDER, filename)
    cv2.imwrite(filepath, image)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, angle])

def main():
    robot = Car()
    driver = Driver()
    keyboard = Keyboard()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display = Display("display_image")

    # Inicia velocidad en 0
    speed = 0.0
    steering_angle = 0.0
    max_speed = 50
    last_angle = 0
    last_capture_time = time.time()
    image_counter =  0

    keyboard.enable(timestep)
    driver.setCruisingSpeed(13.00)

    while robot.step() != -1:
        key = keyboard.getKey()
        steering_angle = 0.0  # dirección recta por defecto en cada frame
        while key != -1:
            if key == Keyboard.UP:
                speed = max_speed
            elif key == Keyboard.DOWN:
                speed = 0.0
            elif key == Keyboard.LEFT:
                 steering_angle = -0.2  # Gira a la izquierda
            elif key == Keyboard.RIGHT:
                steering_angle = 0.2   # Gira a la derecha
            key = keyboard.getKey()

        driver.setCruisingSpeed(speed)
        driver.setSteeringAngle(steering_angle)

        image = get_image(camera)
        angle, lines = detect_lane_angle_trapezoid(image, last_angle)
        last_angle = angle

        display_image_with_lines(display, image, lines)

        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            save_image_and_log(image, steering_angle, image_counter)
            image_counter += 1
            last_capture_time = current_time

if __name__ == "__main__":
    main()
