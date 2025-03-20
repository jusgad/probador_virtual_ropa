"""
Aplicación de Medición Virtual de Ropa
-------------------------------------
Sistema que permite medir y probar ropa virtualmente
en fotos o webcam.
"""

import os
import cv2
import numpy as np
import argparse
import logging
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Importar componentes del sistema
from core.body_detector import BodyDetector
from core.measurement import MeasurementCalculator
from core.clothing_fitter import ClothingFitter
from core.size_recommender import SizeRecommender
from database.db import init_db, db_session
from database.repositories import save_measurement, get_user_measurements
from utils.image_utils import load_image, save_image

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear la aplicación Flask
app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)

# Cargar configuración
CONFIG_PATH = 'config.json'
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
else:
    logger.warning(f"Archivo de configuración {CONFIG_PATH} no encontrado, usando valores por defecto")
    config = {
        "reference_height_cm": 170.0,
        "camera_id": 0,
        "image_size": [640, 480],
        "debug": False
    }

# Inicializar componentes
detector = BodyDetector()
measurement_calc = MeasurementCalculator(reference_height_cm=config.get("reference_height_cm", 170.0))
clothing_fitter = ClothingFitter()
size_recommender = SizeRecommender()

# Asegurarse de que existan los directorios necesarios
os.makedirs('data/users', exist_ok=True)
os.makedirs('data/clothes/shirts', exist_ok=True)
os.makedirs('data/clothes/pants', exist_ok=True)

@app.before_first_request
def initialize():
    """Inicializa la base de datos antes de la primera petición."""
    init_db()

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Cierra la sesión de base de datos al finalizar cada petición."""
    db_session.remove()

@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

@app.route('/measure', methods=['GET', 'POST'])
def measure():
    """Página para realizar mediciones."""
    if request.method == 'POST':
        # Procesar medición desde formulario o cámara
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Guardar imagen subida temporalmente
                img_path = 'static/uploads/temp.jpg'
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                file.save(img_path)
                
                # Procesar imagen
                img = cv2.imread(img_path)
                return process_image(img, img_path)
        
        # Si no hay archivo pero es POST, asumir webcam
        return render_template('measurement.html', mode='webcam')
            
    # GET request - mostrar formulario
    return render_template('measurement.html', mode='upload')

@app.route('/api/measure_webcam', methods=['POST'])
def measure_webcam():
    """API para procesar imágenes de webcam."""
    if 'image' in request.files:
        file = request.files['image']
        img_path = 'static/uploads/webcam.jpg'
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        file.save(img_path)
        
        # Procesar imagen
        img = cv2.imread(img_path)
        result = process_image(img, img_path, return_json=True)
        return jsonify(result)
    
    return jsonify({"error": "No image received"}), 400

@app.route('/results/<measurement_id>')
def results(measurement_id):
    """Muestra resultados de una medición."""
    # Obtener medición de la base de datos
    measurement = get_user_measurements(measurement_id)
    if not measurement:
        return redirect(url_for('index'))
    
    # Obtener recomendaciones de talla
    shirt_size = size_recommender.recommend_size('shirt', measurement)
    pants_size = size_recommender.recommend_size('pants', measurement)
    
    return render_template(
        'results.html',
        measurement=measurement,
        shirt_size=shirt_size,
        pants_size=pants_size
    )

def process_image(img, img_path, return_json=False):
    """
    Procesa una imagen para detección corporal y medidas.
    
    Args:
        img: Imagen cargada con OpenCV
        img_path: Ruta de la imagen
        return_json: Si es True, devuelve JSON en lugar de redireccionar
        
    Returns:
        Redirección a resultados o datos JSON según return_json
    """
    # Detectar cuerpo
    body_landmarks = detector.detect_body(img)
    
    if body_landmarks:
        # Calcular medidas
        measurements = measurement_calc.calculate_measurements(img, body_landmarks)
        
        # Visualizar medidas en la imagen
        result_img = measurement_calc.visualize_measurements(img.copy(), measurements)
        
        # Guardar imagen con medidas
        result_path = 'static/results/measured.jpg'
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        cv2.imwrite(result_path, result_img)
        
        # Guardar medidas en la base de datos
        measurement_id = save_measurement(
            user_id=1,  # Usuario por defecto
            measurements=measurements.to_dict(),
            source_image_path=img_path
        )
        
        if return_json:
            return {
                "success": True,
                "measurement_id": measurement_id,
                "measurements": measurements.to_dict(),
                "result_image": result_path
            }
        else:
            return redirect(url_for('results', measurement_id=measurement_id))
    else:
        if return_json:
            return {"error": "No se detectó un cuerpo en la imagen"}, 400
        else:
            return render_template('measurement.html', error="No se detectó un cuerpo en la imagen")

def run_cli():
    """Ejecuta el programa en modo línea de comandos."""
    parser = argparse.ArgumentParser(description='Medición Virtual de Ropa')
    parser.add_argument('--mode', choices=['webcam', 'file'], default='webcam', help='Modo de captura')
    parser.add_argument('--input', help='Ruta a imagen de entrada (para modo file)')
    parser.add_argument('--clothing', help='Ruta a imagen de prenda')
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        cap = cv2.VideoCapture(config.get("camera_id", 0))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Error leyendo webcam")
                break
                
            # Mostrar instrucciones
            cv2.putText(frame, "Presiona 'c' para capturar o 'q' para salir", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow('Medicion Virtual', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Procesar frame capturado
                img_path = 'static/uploads/capture.jpg'
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                cv2.imwrite(img_path, frame)
                
                body_landmarks = detector.detect_body(frame)
                
                if body_landmarks:
                    # Calcular medidas
                    measurements = measurement_calc.calculate_measurements(frame, body_landmarks)
                    
                    # Visualizar medidas
                    result_img = measurement_calc.visualize_measurements(frame.copy(), measurements)
                    
                    # Mostrar resultados
                    cv2.imshow('Resultados', result_img)
                    
                    # Imprimir medidas
                    print("\nMedidas calculadas:")
                    for key, value in measurements.__dict__.items():
                        if key != 'pixel_to_cm_ratio' and value is not None:
                            print(f"{key}: {value:.1f} cm")
                    
                    # Si se especificó una prenda, probarla
                    if args.clothing and os.path.exists(args.clothing):
                        clothing_img = cv2.imread(args.clothing, cv2.IMREAD_UNCHANGED)
                        if clothing_img is not None:
                            clothing_type = os.path.basename(args.clothing).split('_')[0]
                            
                            # Recomendar talla
                            size = size_recommender.recommend_size(clothing_type, measurements)
                            print(f"\nTalla recomendada para {clothing_type}: {size}")
                            
                            # Ajustar prenda al cuerpo
                            fitted_img = clothing_fitter.fit_clothing(
                                frame.copy(), clothing_img, body_landmarks, measurements
                            )
                            
                            # Mostrar resultado
                            cv2.imshow('Prueba Virtual', fitted_img)
                    
                    # Esperar tecla antes de continuar
                    cv2.waitKey(0)
                else:
                    print("No se detectó un cuerpo en la imagen")
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif args.mode == 'file' and args.input:
        if not os.path.exists(args.input):
            logger.error(f"Archivo no encontrado: {args.input}")
            return
            
        # Cargar imagen
        img = cv2.imread(args.input)
        if img is None:
            logger.error("Error al cargar la imagen")
            return
            
        # Procesar imagen
        body_landmarks = detector.detect_body(img)
        
        if body_landmarks:
            # Calcular medidas
            measurements = measurement_calc.calculate_measurements(img, body_landmarks)
            
            # Visualizar medidas
            result_img = measurement_calc.visualize_measurements(img.copy(), measurements)
            
            # Mostrar resultados
            cv2.imshow('Resultados', result_img)
            
            # Imprimir medidas
            print("\nMedidas calculadas:")
            for key, value in measurements.__dict__.items():
                if key != 'pixel_to_cm_ratio' and value is not None:
                    print(f"{key}: {value:.1f} cm")
            
            # Si se especificó una prenda, probarla
            if args.clothing and os.path.exists(args.clothing):
                clothing_img = cv2.imread(args.clothing, cv2.IMREAD_UNCHANGED)
                if clothing_img is not None:
                    clothing_type = os.path.basename(args.clothing).split('_')[0]
                    
                    # Recomendar talla
                    size = size_recommender.recommend_size(clothing_type, measurements)
                    print(f"\nTalla recomendada para {clothing_type}: {size}")
                    
                    # Ajustar prenda al cuerpo
                    fitted_img = clothing_fitter.fit_clothing(
                        img.copy(), clothing_img, body_landmarks, measurements
                    )
                    
                    # Mostrar resultado
                    cv2.imshow('Prueba Virtual', fitted_img)
            
            # Esperar tecla antes de salir
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No se detectó un cuerpo en la imagen")

if __name__ == "__main__":
    # Para ejecutar la app web, usar este bloque
    app.run(debug=config.get("debug", False), host='0.0.0.0', port=5000)
    
    # Para ejecutar en modo CLI, comentar lo anterior y descomentar esta línea
    # run_cli()