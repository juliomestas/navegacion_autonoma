from controller import Robot, Camera, Keyboard
from vehicle import Driver
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model

# =======================
# CARGAR MAPA ÁNGULO-VELOCIDAD
# =======================
try:
    df_map = pd.read_csv("steering_to_speed_map.csv")
    steering_speed_dict = dict(zip(df_map['rounded_steering'], df_map['speed']))
except Exception as e:
    print(f"Error al cargar steering_to_speed_map.csv: {e}")
    steering_speed_dict = {}

# Función para obtener velocidad recomendada desde el diccionario (máx. 80 km/h)
def get_speed_from_steering(angle):
    rounded_angle = round(angle, 1)
    return min(steering_speed_dict.get(rounded_angle, 30.0), 80.0)

# =======================
# INICIALIZACIÓN DE DISPOSITIVOS
# =======================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)

driver = Driver()
cruising_speed = 30.0
driver.setCruisingSpeed(cruising_speed)

# =======================
# CARGAR MODELO
# =======================
try:
    model = load_model("model_balanceado_maxpool_gray_blur.h5", compile=False)
except Exception as e:
    print(f"Error cargando modelo: {e}")
    exit()

# =======================
# PREPROCESAMIENTO
# =======================
def preprocess(img):
    img = np.frombuffer(img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))[:, :, :3].copy()
    height = img.shape[0]
    img[:int(height * 0.3), :] = 0  # Recorte parte superior
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (200, 66))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)      # (66,200) → (66,200,1)
    return img.reshape(1, 66, 200, 1)

# =======================
# LOOP PRINCIPAL
# =======================
while robot.step() != -1:
    img_raw = camera.getImage()
    img_processed = preprocess(img_raw)

    try:
        steering_angle = float(model.predict(img_processed, verbose=0)[0][0])
    except Exception as e:
        print(f"Error al predecir: {e}")
        steering_angle = 0.0

    cruising_speed = get_speed_from_steering(steering_angle)

    key = keyboard.getKey()
    if key == Keyboard.UP:
        cruising_speed = min(cruising_speed + 5.0, 50.0)
    elif key == Keyboard.DOWN:
        cruising_speed = max(0.0, cruising_speed - 5.0)

    driver.setSteeringAngle(steering_angle)
    driver.setCruisingSpeed(cruising_speed)

    # ✅ IMPRIMIR SIEMPRE EN CADA ITERACIÓN
    print(f"Ángulo: {steering_angle:.3f} rad | Velocidad: {cruising_speed:.1f} km/h")