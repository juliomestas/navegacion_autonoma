from controller import Display, Robot, Camera, Keyboard, Radar # Importa Radar
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

# --- Nuevos ajustes para Radar ---
RADAR_NAME = "front_radar"  
                            
BRAKING_DISTANCE = 25.0      # Distancia en metros a la que el coche frenará
MIN_RADAR_RANGE = 0.2       # El rango mínimo del Radar
COLLISION_SPEED_REDUCTION = 0.0 # Velocidad a la que el coche debe reducir cuando detecta un obstáculo

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

    display = robot.getDevice("display_image")

    # --- Inicialización del Radar ---
    radar = robot.getDevice(RADAR_NAME)
    radar.enable(timestep)
    radar_max_range = radar.getMaxRange()

    current_speed = 0.0
    steering_angle = 0.0
    max_speed = 30 # 
    last_angle = 0
    last_capture_time = time.time()
    image_counter = 0
    
    is_braking = False

    keyboard.enable(timestep)
    # driver.setCruisingSpeed(13.00) # Comentado para que la velocidad sea controlada por la lógica

    while robot.step() != -1:
        
        # --- Lectura y Procesamiento del Radar ---
        radar_targets = radar.getTargets()
        
        min_distance = radar_max_range #50.0 como anterior
        obstacle_detected_for_braking = False

        if radar_targets:
            for target in radar_targets:
                distance = target.distance
                
                if distance < min_distance and distance > MIN_RADAR_RANGE:
                    min_distance = distance
        
        # --- Lógica de frenado por Radar (Prioritaria) ---
        if min_distance < BRAKING_DISTANCE:
            obstacle_detected_for_braking = True
        
        # --- Depuración de Radar ---
        print(f"Distancia mínima detectada por Radar: {min_distance:.2f} m. Obstáculo para frenado: {obstacle_detected_for_braking}")

        # --- Control de Velocidad y Frenado ---
        if obstacle_detected_for_braking:
            current_speed = COLLISION_SPEED_REDUCTION # Detener completamente
            is_braking = True
            print("¡Peatón detectado DENTRO DEL UMBRAL! Frenando agresivamente...")
        else:
            # Si no hay obstáculo, y NO estábamos frenando por detección de obstáculo,
            # PERMITE EL CONTROL NORMAL POR TECLADO
            if is_braking: # Si antes estaba frenando, ahora puede volver a moverse (pero empezando desde 0)
                current_speed = COLLISION_SPEED_REDUCTION # Mantiene velocidad en 0 hasta que el usuario acelere de nuevo
                is_braking = False
        # --- Control por teclado (SOLO SI NO HAY DETECCIÓN ACTIVA DE OBSTÁCULO Y NO ESTAMOS EN ESTADO DE FRENADO) ---
        # El bucle while para el teclado debe estar dentro de la lógica 'else'
        # para que la detección de Radar tenga prioridad.
        key = keyboard.getKey()
        # Reiniciar steering_angle en cada frame si no se presiona una tecla de dirección
        steering_angle = 0.0

        while key != -1:
            if key == Keyboard.UP:
                current_speed = max_speed
            elif key == Keyboard.DOWN:
                current_speed = 0.0
            elif key == Keyboard.LEFT:
                steering_angle = -0.2
            elif key == Keyboard.RIGHT:
                steering_angle = 0.2
            key = keyboard.getKey()
        
        # --- Imprimir la velocidad actual del vehículo ---
        print(f"Velocidad actual: {driver.getCurrentSpeed():.2f} m/s") # Añade esta línea
        
        # --- Aplica la velocidad y el ángulo de dirección al Driver ---
        driver.setCruisingSpeed(current_speed)
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