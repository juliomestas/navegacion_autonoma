from controller import Robot, Camera, Keyboard, Display
from vehicle import Driver
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
import os

# =======================
# CONFIGURACIÓN
# =======================
IMG_WIDTH, IMG_HEIGHT = 200, 66  # Para el modelo
DISPLAY_WIDTH, DISPLAY_HEIGHT = 320, 160  # Solo para visualización

# =======================
# CARGAR MAPA ÁNGULO-VELOCIDAD
# =======================
try:
    df_map = pd.read_csv("steering_to_speed_map.csv")
    steering_speed_dict = dict(zip(df_map['rounded_steering'], df_map['speed']))
except Exception as e:
    print(f"❌ Error al cargar mapa velocidad: {e}")
    steering_speed_dict = {}

def get_speed_from_steering(angle):
    rounded = round(angle, 1)
    return min(steering_speed_dict.get(rounded, 30.0), 80.0)

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

display = robot.getDevice("display")
if not display:
    print("⚠️ Display no encontrado.")

# =======================
# CARGAR MODELO
# =======================
try:
    model = load_model("model_balanceado_maxpool_gray_blur.h5", compile=False)
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
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
    # Convertir raw de cámara a imagen RGB
    img = np.frombuffer(img_raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))[:, :, :3].copy()
    
    # Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar máscara
    masked = apply_trapezoid_mask(gray)

    # Desenfocar y normalizar
    blurred = cv2.GaussianBlur(masked, (5, 5), 0)
    input_img = cv2.resize(blurred, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
    input_img = input_img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)  # (1, 66, 200, 1)

    # Para visualización
    vis_img = cv2.resize(blurred, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    rgba = cv2.cvtColor(cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGBA)

    return input_img, rgba, blurred

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
        print(f"❌ Error de predicción: {e}")
        steering_angle = 0.0

    # Velocidad basada en el ángulo
    speed = get_speed_from_steering(steering_angle)

    # Ajuste manual con teclado
    key = keyboard.getKey()
    if key == Keyboard.UP:
        speed = min(speed + 5.0, 50.0)
    elif key == Keyboard.DOWN:
        speed = max(speed - 5.0, 0.0)

    driver.setSteeringAngle(steering_angle)
    driver.setCruisingSpeed(speed)

    print(f"Ángulo: {steering_angle:.3f} rad | Velocidad: {speed:.1f} km/h")

    # Mostrar en Display
    try:
        image_id = display.imageNew(display_img.tobytes(), Display.RGBA, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        display.imagePaste(image_id, 0, 0, False)
        display.imageDelete(image_id)
    except Exception as e:
        print(f"⚠️ Error en display: {e}")

    # Guardar imagen cada 10 pasos
    if step_counter % 10 == 0:
        debug_frame = cv2.resize(raw_gray, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        filename = f"debug_input/frame_{step_counter:04d}_angle_{steering_angle:+.2f}.png"
        cv2.imwrite(filename, debug_frame)

    step_counter += 1