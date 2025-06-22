from controller import Robot, Camera, Keyboard, Radar
from vehicle import Driver
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time

# --- Parámetros del radar ---
RADAR_NAME = "front_radar"
BRAKING_DISTANCE = 25.0     # distancia a la que frena completamente
FOLLOW_DISTANCE = 10.0      # distancia en la que empieza a acelerar suavemente
MIN_RADAR_RANGE = 0.2
MAX_SPEED = 40.0            # km/h
COLLISION_SPEED_REDUCTION = 0.0

# --- Inicialización del entorno ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)

driver = Driver()

radar = robot.getDevice(RADAR_NAME)
radar.enable(timestep)
radar_max_range = radar.getMaxRange()

# --- Cargar el modelo entrenado ---
try:
    model = load_model("model_balanceado_maxpool_gray_blur.h5", compile=False)
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    exit()

# --- Preprocesamiento de imagen ---
def preprocess(img):
    img = np.frombuffer(img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))[:, :, :3].copy()
    height = img.shape[0]
    img[:int(height * 0.3), :] = 0  # Recorte superior
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (200, 66))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img.reshape(1, 66, 200, 1)

# --- Mapeo de distancia a velocidad (lineal) ---
def calcular_velocidad_por_distancia(distancia):
    if distancia <= FOLLOW_DISTANCE:
        return 5.0  # mínimo avance para no quedarse detenido
    elif distancia >= BRAKING_DISTANCE:
        return MAX_SPEED
    else:
        # Escalado lineal entre FOLLOW_DISTANCE y BRAKING_DISTANCE
        return 5.0 + (MAX_SPEED - 5.0) * ((distancia - FOLLOW_DISTANCE) / (BRAKING_DISTANCE - FOLLOW_DISTANCE))

# --- Bucle principal ---
current_speed = MAX_SPEED
is_braking = False

while robot.step() != -1:
    # Obtener imagen y predecir ángulo
    img_raw = camera.getImage()
    img_processed = preprocess(img_raw)
    try:
        steering_angle = float(model.predict(img_processed, verbose=0)[0][0])
    except:
        steering_angle = 0.0

    # Lectura de radar
    radar_targets = radar.getTargets()
    min_distance = radar_max_range
    obstacle_detected = False

    if radar_targets:
        for target in radar_targets:
            if MIN_RADAR_RANGE < target.distance < min_distance:
                min_distance = target.distance

    # Lógica de control de velocidad
    if min_distance < FOLLOW_DISTANCE:
        current_speed = COLLISION_SPEED_REDUCTION
        is_braking = True
        print("¡Obstáculo muy cerca! Frenando...")
    elif min_distance < BRAKING_DISTANCE:
        current_speed = calcular_velocidad_por_distancia(min_distance)
        is_braking = False
        print(f"Obstáculo cerca. Ajustando velocidad a {current_speed:.1f} km/h")
    else:
        current_speed = MAX_SPEED
        is_braking = False

    # Control por teclado (solo UP/DOWN para modificar target general de velocidad)
    key = keyboard.getKey()
    while key != -1:
        if key == Keyboard.UP:
            MAX_SPEED = min(MAX_SPEED + 5.0, 60.0)
        elif key == Keyboard.DOWN:
            MAX_SPEED = max(5.0, MAX_SPEED - 5.0)
        key = keyboard.getKey()

    # Aplicar control al vehículo
    driver.setSteeringAngle(steering_angle)
    driver.setCruisingSpeed(current_speed)

    print(f"Ángulo: {steering_angle:.3f} rad | Velocidad ajustada: {current_speed:.1f} km/h | Distancia radar: {min_distance:.2f} m")