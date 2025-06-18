from controller import Camera, Keyboard
from vehicle import Car, Driver
import numpy as np
import cv2
import os
import csv
import time

# Ajustes
CAPTURE_INTERVAL = 0.2
IMAGES_FOLDER = "captured_images"
CSV_FILE = "angles.csv"

if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "steering_angle", "vision_angle", "speed"])  # 👈 nueva columna

def apply_trapezoid_mask(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    vertices = np.array([[ 
        (int(width * 0.05), height),
        (int(width * 0.4), int(height * 0.5)),
        (int(width * 0.6), int(height * 0.5)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, (255, 255, 255))
    return cv2.bitwise_and(image, mask)

def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    rgb_image = image[:, :, :3].copy()
    height = rgb_image.shape[0]
    cutoff = int(height * 0.3)
    rgb_image[:cutoff, :] = 0
    return apply_trapezoid_mask(rgb_image)

def greyscale_cv2(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_lane_angle_trapezoid(image, last_angle):
    gray = greyscale_cv2(image)
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
    return max(-0.5, min(0.5, angle)), lines

# ✅ Guardar también velocidad
def save_image_and_log(image, steering_angle, vision_angle, speed, count):
    filename = f"img_{count:05}.png"
    filepath = os.path.join(IMAGES_FOLDER, filename)
    cv2.imwrite(filepath, image)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, steering_angle, vision_angle, speed])  # 👈 incluir speed

def main():
    robot = Car()
    driver = Driver()
    keyboard = Keyboard()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    speed = 0.0
    max_speed = 80.0
    steering_angle = 0.0
    last_angle = 0.0
    last_capture_time = time.time()
    image_counter = 0
    mode = "MANUAL"

    keyboard.enable(timestep)

    while robot.step() != -1:
        key = keyboard.getKey()
        keys_pressed = set()
        while key != -1:
            keys_pressed.add(key)
            key = keyboard.getKey()

        # Cambiar modo con teclas A o M
        if ord('A') in keys_pressed:
            mode = 'AUTO'
        elif ord('M') in keys_pressed:
            mode = 'MANUAL'

        # Control de velocidad acumulativa
        if Keyboard.UP in keys_pressed:
            speed = min(speed + 5.0, max_speed)
        elif Keyboard.DOWN in keys_pressed:
            speed = max(speed - 5.0, 0.0)

        image = get_image(camera)
        angle, _ = detect_lane_angle_trapezoid(image, last_angle)
        last_angle = angle

        if mode == 'AUTO':
            steering_angle = angle * 2.5
        elif mode == 'MANUAL':
            if Keyboard.LEFT in keys_pressed:
                steering_angle -= 0.01
            elif Keyboard.RIGHT in keys_pressed:
                steering_angle += 0.01
            else:
                steering_angle = 0.0
        else:
            steering_angle = 0.0

        steering_angle = max(-0.2, min(0.2, steering_angle))

        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(speed)

        print(f"[{mode}] Speed: {speed:.1f}, Steering: {steering_angle:.2f}, Vision: {angle:.2f}")

        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            save_image_and_log(image, steering_angle, angle, speed, image_counter)  # 👈 incluir speed
            image_counter += 1
            last_capture_time = current_time

if __name__ == "__main__":
    main()