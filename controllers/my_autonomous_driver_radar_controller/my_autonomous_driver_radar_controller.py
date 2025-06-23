from controller import Robot, Camera, Keyboard, Display, Radar
from vehicle import Driver
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
import os

# =======================
# CONFIGURACIÓN
# =======================
IMG_WIDTH, IMG_HEIGHT = 200, 66
DISPLAY_WIDTH, DISPLAY_HEIGHT = 320, 160
RADAR_NAME = "front_radar"
BRAKING_DISTANCE = 25.0
FOLLOW_DISTANCE = 10.0
MIN_RADAR_RANGE = 1.0
MAX_SPEED = 40.0
COLLISION_SPEED_REDUCTION = 0.0

# =======================
# MAPEO DE ANGULO A VELOCIDAD
# =======================
try:
    df_map = pd.read_csv("steering_to_speed_map.csv")
    steering_speed_dict = dict(zip(df_map['rounded_steering'], df_map['speed']))
except Exception as e:
    print(f"Error al cargar mapa velocidad: {e}")
    steering_speed_dict = {}

def get_speed_from_steering(angle):
    rounded = round(angle, 1)
    return min(steering_speed_dict.get(rounded, 30.0), MAX_SPEED)

# =======================
# INICIALIZACIÓN DE WEBOTS
# =======================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)

driver = Driver()
driver.setCruisingSpeed(30.0)

display = robot.getDevice("display_image")
if not display:
    print("Display no encontrado.")

radar = robot.getDevice(RADAR_NAME)
radar.enable(timestep)
radar_max_range = radar.getMaxRange()

# =======================
# CARGAR MODELO
# =======================
try:
    model = load_model("model_balanceado_maxpool_gray_blur.h5", compile=False)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# =======================
# FUNCIONES AUXILIARES
# =======================
def apply_trapezoid_mask(image):
    h, w = image.shape[:2]
    mask = np.zeros_like(image)
    vertices = np.array([[
        (int(w * 0.1), h),
        (int(w * 0.4), int(h * 0.4)),
        (int(w * 0.6), int(h * 0.4)),
        (int(w * 0.9), h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)

def preprocess(img_raw):
    img = np.frombuffer(img_raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))[:, :, :3].copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked = apply_trapezoid_mask(gray)
    blurred = cv2.GaussianBlur(masked, (5, 5), 0)
    input_img = cv2.resize(blurred, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
    input_img = input_img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
    vis_img = cv2.resize(blurred, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    rgba = cv2.cvtColor(cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGBA)
    return input_img, rgba, blurred

def calcular_velocidad_por_distancia(distancia):
    if distancia <= FOLLOW_DISTANCE:
        return 5.0
    elif distancia >= BRAKING_DISTANCE:
        return MAX_SPEED
    else:
        return 5.0 + (MAX_SPEED - 5.0) * ((distancia - FOLLOW_DISTANCE) / (BRAKING_DISTANCE - FOLLOW_DISTANCE))

# =======================
# LOOP PRINCIPAL
# =======================
step_counter = 0
os.makedirs("webots_frames", exist_ok=True)
os.makedirs("debug_input", exist_ok=True)

while robot.step() != -1:
    raw = camera.getImage()
    input_model, display_img, raw_gray = preprocess(raw)

    try:
        prediction = model.predict(input_model, verbose=0)
        steering_angle = float(prediction[0][0])
        print(f"⏎ Predicción cruda: {prediction}")
    except Exception as e:
        print(f"Error de predicción: {e}")
        steering_angle = 0.0

    # Velocidad basada en el ángulo
    base_speed = get_speed_from_steering(steering_angle)
    current_speed = base_speed

    # Evaluar radar
    radar_targets = radar.getTargets()
    min_distance = radar_max_range
    for target in radar_targets:
        if MIN_RADAR_RANGE < target.distance < min_distance:
            min_distance = target.distance

    if min_distance < BRAKING_DISTANCE:
        radar_speed = calcular_velocidad_por_distancia(min_distance)
        current_speed = min(base_speed, radar_speed)

    # Teclado
    key = keyboard.getKey()
    if key == Keyboard.UP:
        current_speed = min(current_speed + 5.0, MAX_SPEED)
    elif key == Keyboard.DOWN:
        current_speed = max(current_speed - 5.0, 0.0)

    driver.setSteeringAngle(steering_angle)
    driver.setCruisingSpeed(current_speed)

    print(f"Ángulo: {steering_angle:.3f} rad | Velocidad: {current_speed:.1f} km/h | Dist. Radar: {min_distance:.2f} m")

    try:
        image_id = display.imageNew(display_img.tobytes(), Display.RGBA, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        display.imagePaste(image_id, 0, 0, False)
        display.imageDelete(image_id)
    except Exception as e:
        print(f"Error en display: {e}")

    if step_counter % 10 == 0:
        debug_frame = cv2.resize(raw_gray, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        filename = f"debug_input/frame_{step_counter:04d}_angle_{steering_angle:+.2f}.png"
        cv2.imwrite(filename, debug_frame)

    step_counter += 1
