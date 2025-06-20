from controller import Robot, Camera, Keyboard
from vehicle import Driver
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Inicialización del robot y timestep
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Inicializar cámara
camera = robot.getDevice("camera")
camera.enable(timestep)

# Inicializar teclado
keyboard = Keyboard()
keyboard.enable(timestep)

# Inicializar controlador del vehículo
driver = Driver()
cruising_speed = 40.0  # velocidad inicial en km/h
driver.setCruisingSpeed(cruising_speed)

# Cargar modelo entrenado
try:
    model = load_model("model_circuito.h5", compile=False)
except Exception as e:
    print(f"Error cargando modelo: {e}")
    exit()

# Preprocesamiento de imagen
def preprocess(img):
    img = np.frombuffer(img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))[:, :, :3].copy()
    height = img.shape[0]
    img[:int(height * 0.3), :] = 0  # Recorte superior
    img = cv2.resize(img, (200, 66))
    img = img.astype(np.float32) / 255.0
    return img.reshape(1, 66, 200, 3)

# Bucle principal del controlador
while robot.step() != -1:
    img_raw = camera.getImage()
    img_processed = preprocess(img_raw)

    # Predicción del ángulo de giro
    try:
        steering_angle = float(model.predict(img_processed, verbose=0)[0][0])
    except Exception as e:
        print(f"Error al predecir: {e}")
        steering_angle = 0.0

    # Imprimir valores
    print(f"Ángulo de giro: {steering_angle:.3f} rad | Velocidad: {cruising_speed:.1f} km/h")

    # Aplicar ángulo al vehículo
    driver.setSteeringAngle(steering_angle)

    # Leer entrada de teclado para modificar velocidad
    key = keyboard.getKey()
    if key == Keyboard.UP:
        cruising_speed += 5.0
        driver.setCruisingSpeed(cruising_speed)
    elif key == Keyboard.DOWN:
        cruising_speed = max(0.0, cruising_speed - 5.0)
        driver.setCruisingSpeed(cruising_speed)