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
MIN_RADAR_RANGE = 1.0       # El rango mínimo del Radar
MAX_SPEED = 40.0            # km/h
COLLISION_SPEED_REDUCTION = 0.0

# --- Parámetros de giro por teclado ---
KEYBOARD_STEERING_ANGLE = 0.2 # Valor en radianes (ajusta según necesites)

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
    # Obtener imagen y predecir ángulo (valor inicial por IA)
    img_raw = camera.getImage()
    img_processed = preprocess(img_raw)
    try:
        # Predicción del ángulo por el modelo (este será el valor por defecto)
        ai_steering_angle = float(model.predict(img_processed, verbose=0)[0][0])
    except Exception as e: # Captura la excepción para depuración
        print(f"Error en la predicción del modelo: {e}")
        ai_steering_angle = 0.0
    
    # Inicializa steering_angle con el valor predicho por la IA
    steering_angle = ai_steering_angle

    # --- Lectura y Procesamiento del Radar ---
    radar_targets = radar.getTargets()
    
    # --- CORRECCIÓN: Inicializar min_distance_relevant antes de usarlo ---
    min_distance_relevant = radar_max_range 
    obstacle_detected_for_action = False 

    if radar_targets:
        # Depuración (descomenta si necesitas ver todos los targets)
        # print(f"--- Targets detectados ({len(radar_targets)}) ---")
        for target in radar_targets:
            distance = target.distance
            
            # --- NOTA IMPORTANTE: Tu radar no soporta azimuth ni velocity ---
            # Las líneas para azimuth y velocity se eliminan de aquí
            # porque tu modelo de radar no las proporciona, según los errores anteriores.
            # NO intentar acceder a target.azimuth o target.velocity aquí.
            
            # Opcional: Imprimir cada target para depuración, pero solo con 'distance'
            # print(f"  Target ID: {target.id}, Dist: {distance:.2f}m")

            # --- Lógica de filtrado de objetivos relevantes (solo por distancia) ---
            # Si tu radar no tiene azimuth/velocity, la lógica se simplifica a la distancia más cercana
            # que está dentro de tu rango de interés y es mayor que MIN_RADAR_RANGE.
            
            if MIN_RADAR_RANGE < distance < min_distance_relevant:
                min_distance_relevant = distance
                obstacle_detected_for_action = True # Marcamos que hay un obstáculo relevante
                
    # --- Lógica de control de velocidad (prioridad del radar) ---
    # Ahora usamos min_distance_relevant y obstacle_detected_for_action
    if obstacle_detected_for_action: # Solo ajustamos velocidad si un obstáculo relevante fue detectado
        if min_distance_relevant < FOLLOW_DISTANCE:
            current_speed = COLLISION_SPEED_REDUCTION # Frena completamente
            is_braking = True
            print("¡Obstáculo muy cerca! Frenando...")
        elif min_distance_relevant < BRAKING_DISTANCE:
            current_speed = calcular_velocidad_por_distancia(min_distance_relevant) # Ajusta velocidad gradualmente
            is_braking = False # No está frenando a 0, sino ajustando
            print(f"Obstáculo cerca. Ajustando velocidad a {current_speed:.1f} km/h")
    else: # Si NO se detectó ningún obstáculo relevante, el coche puede ir a MAX_SPEED
        current_speed = MAX_SPEED
        is_braking = False # Reiniciar bandera de frenado si ya no hay obstáculo o está lejos
        
    # --- Control por teclado (prioridad sobre la IA y la lógica de velocidad si hay tecla presionada) ---
    key = keyboard.getKey()
    
    # Este bucle while key != -1: debe estar al final de la lógica para que las pulsaciones
    # de teclado sobrescriban las decisiones del radar/IA en el mismo timestep.
    # No mover la lógica del radar o IA DENTRO de este 'while key != -1', sino gestionarla
    # ANTES y permitir que el teclado sobrescriba AL FINAL del paso.
    
    while key != -1: 
        # Control de Velocidad
        if key == Keyboard.UP:
            MAX_SPEED = min(MAX_SPEED + 5.0, 60.0) # Aumenta el límite de velocidad configurable
            # Al cambiar MAX_SPEED, también actualiza current_speed si no hay obstáculo
            if not obstacle_detected_for_action:
                 current_speed = MAX_SPEED 
        elif key == Keyboard.DOWN:
            MAX_SPEED = max(5.0, MAX_SPEED - 5.0) # Disminuye el límite de velocidad configurable
            # Al cambiar MAX_SPEED, también actualiza current_speed si no hay obstáculo
            if not obstacle_detected_for_action:
                current_speed = MAX_SPEED 
        
        # Control de Giros
        elif key == Keyboard.LEFT:
            steering_angle = -KEYBOARD_STEERING_ANGLE # Gira a la izquierda
        elif key == Keyboard.RIGHT:
            steering_angle = KEYBOARD_STEERING_ANGLE # Gira a la derecha
            
        key = keyboard.getKey() # Lee la siguiente tecla (si hay)

    # --- Aplicar control al vehículo ---
    driver.setSteeringAngle(steering_angle) # Aplicamos el ángulo (predicho por IA o sobrescrito por teclado)
    driver.setCruisingSpeed(current_speed)

    # --- Impresiones para depuración ---
    # Asegúrate de imprimir siempre la velocidad real del driver
    print(f"Ángulo: {steering_angle:.3f} rad | Vel. Ajustada: {current_speed:.1f} km/h | Dist. Radar: {min_distance_relevant:.2f} m | Vel. Real: {driver.getCurrentSpeed():.2f} km/h")

    # --- Captura y log de imágenes (mantienes esto al final) ---
    # Aquí puedes usar 'steering_angle' que es el ángulo final aplicado.
    # img_raw ya es la imagen original para guardar.
    current_time = time.time()
    # No necesitas 'last_capture_time' ni 'CAPTURE_INTERVAL' si solo quieres capturar al final del script.
    # Si quieres una captura periódica, sí. Tu código original usa captura periódica, lo mantengo.
    # if current_time - last_capture_time >= CAPTURE_INTERVAL:
    #    save_image_and_log(img_raw, steering_angle, image_counter)
    #    image_counter += 1
    #    last_capture_time = current_time