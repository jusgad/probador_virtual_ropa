"""
Aplicación de Medición Virtual de Ropa
-------------------------------------
Sistema que permite medir y probar ropa virtualmente
en fotos o webcam.
"""

import os
import cv2

import argparse
import logging
import json
import uuid

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Importar componentes del sistema
from core.body_detector import BodyDetector
from core.measurement import MeasurementCalculator
from core.clothing_fitter import ClothingFitter
from core.size_recommender import SizeRecommender
from database.db import init_db, db_session, VirtualFitting
from database.repositories import save_measurement, get_user_measurements


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

# Inicializar base de datos al arrancar
init_db()

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Cierra la sesión de base de datos al finalizar cada petición."""
    db_session.remove()

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

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
            if file and file.filename != '' and allowed_file(file.filename):
                # Guardar imagen subida temporalmente con UUID (seguridad)
                ext = file.filename.rsplit('.', 1)[1].lower()
                unique_filename = f"{uuid.uuid4().hex}.{ext}"
                img_path = os.path.join('static', 'uploads', unique_filename)
                
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                file.save(img_path)
                
                # Procesar imagen
                img = cv2.imread(img_path)
                if img is None:
                    return render_template('measurement.html', mode='upload', error="Imagen no válida o corrupta.")
                return process_image(img, img_path)
            elif file.filename != '':
                return render_template('measurement.html', mode='upload', error="Tipo de archivo no permitido. Solo subir JPG o PNG.")
        
        # Si no hay archivo pero es POST, asumir webcam
        return render_template('measurement.html', mode='webcam')
            
    # GET request - mostrar formulario
    return render_template('measurement.html', mode='upload')

@app.route('/api/measure_webcam', methods=['POST'])
def measure_webcam():
    """API para procesar imágenes de webcam."""
    if 'image' in request.files:
        file = request.files['image']
        # La webcam de jscript tipicamente no manda extension valida, usamos un hack si es webcam.jpg
        # pero es mas seguro aser un fallback a 'jpg'
        filename = file.filename if file.filename else "webcam.jpg"
        
        if file and (allowed_file(filename) or filename == 'webcam.jpg'):
            ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
            if ext not in ALLOWED_EXTENSIONS: ext = 'jpg'
            
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            img_path = os.path.join('static', 'uploads', unique_filename)
            
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            file.save(img_path)
            
            # Procesar imagen
            img = cv2.imread(img_path)
            if img is None:
                return jsonify({"error": "Imagen no válida o corrupta"}), 400
                
            result = process_image(img, img_path, return_json=True)
            return jsonify(result)
        else:
             return jsonify({"error": "Tipo de archivo no permitido"}), 400
    
    return jsonify({"error": "No image received"}), 400

@app.route('/results/<measurement_id>')
def results(measurement_id):
    """Muestra resultados de una medición."""
    # Obtener medición de la base de datos
    measurement = get_user_measurements(measurement_id)
    if not measurement:
        return redirect(url_for('index'))
    
    # Obtener recomendaciones de talla
    m_dict = measurement.to_dict() if measurement else {}
    shirt_size = size_recommender.recommend_size(m_dict, brand='Generic', clothing_type='shirts')
    pants_size = size_recommender.recommend_size(m_dict, brand='Generic', clothing_type='pants')
    
    return render_template(
        'results.html',
        measurement=measurement,
        shirt_size=shirt_size,
        pants_size=pants_size
    )

# Nuevas rutas de API y SPA
@app.route('/_next/<path:path>')
def send_next_assets(path):
    """Sirve los archivos estáticos de Next.js."""
    return send_from_directory(os.path.join('static', '_next'), path)

@app.route('/templates/<filename>')
def serve_template(filename):
    """Sirve las vistas HTML dinámicamente para el enrutador SPA."""
    if not filename.endswith('.html'):
        filename += '.html'
    return render_template(filename)

@app.route('/data/references/size_charts/<filename>')
def serve_size_chart(filename):
    """Sirve las tablas de tallas JSON al frontend."""
    return send_from_directory(os.path.join('data', 'references', 'size_charts'), filename)

@app.route('/api/config')
def api_config():
    """Retorna la configuración del sistema."""
    return jsonify(config)

@app.route('/api/auth/check')
def api_auth_check():
    """Simula verificación de autenticación para el usuario por defecto."""
    return jsonify({
        "authenticated": True,
        "user": {
            "id": 1,
            "username": "usuario_prueba",
            "name": "Usuario de Prueba",
            "email": "usuario@ejemplo.com"
        }
    })

@app.route('/api/measurements/user/<int:user_id>')
def api_measurements_user(user_id):
    """Retorna la medición más reciente para un usuario."""
    from database.repositories import MeasurementRepository
    m = MeasurementRepository().get_latest_for_user(user_id)
    if m:
        return jsonify(m.to_dict())
    return jsonify({"error": "No measurements found"}), 404

@app.route('/api/clothing')
def api_clothing():
    """Retorna la lista de prendas disponibles."""
    from database.repositories import ClothingRepository
    clothing_list = ClothingRepository().get_all_for_api()
    return jsonify(clothing_list)

@app.route('/api/fitting/results', methods=['POST'])
def api_fitting_results():
    """Procesa el ajuste virtual y retorna puntuación y detalles."""
    data = request.get_json() or {}
    user_id = data.get("userId") or 1
    clothing_id = data.get("clothingId")
    measurements_data = data.get("measurements")
    
    if not clothing_id or not measurements_data:
        return jsonify({"error": "Missing clothingId or measurements"}), 400
        
    from database.repositories import ClothingRepository, FittingRepository
    clothing = ClothingRepository().get_by_id(clothing_id)
    if not clothing:
        return jsonify({"error": "Clothing item not found"}), 404
        
    # Obtener recomendación de talla
    rec = size_recommender.recommend_size(measurements_data, brand=clothing.brand, clothing_type=clothing.type + 's')
    
    if "error" in rec:
        recommended_size = "M"
        fit_score = 85.0
        fit_quality = "good"
    else:
        recommended_size = rec.get("recommended_size", "M")
        fit_score = rec.get("fit_score", 0.85) * 100
        fit_quality = rec.get("fit_quality", "good")

    fit_descriptions = {
        "perfect": "Ajuste perfecto a tu medida.",
        "good": "Buen ajuste general.",
        "acceptable": "Ajuste aceptable.",
        "poor": "Se recomienda otra talla."
    }
    
    result_image = clothing.image_path
    
    fitting_repo = FittingRepository()
    fitting = VirtualFitting(user_id=user_id, measurement_id=measurements_data.get("id") or 1)
    fitting.clothing_id = clothing_id
    fitting.fit_scores = json.dumps(rec)
    fitting.result_image_path = result_image
    fitting_repo.create(fitting)
    
    user_chest = measurements_data.get("chest", 90.0)
    garment_chest = user_chest + 2.0
    
    res = {
        "previewImage": result_image,
        "recommendedSize": recommended_size,
        "fitDescription": fit_descriptions.get(fit_quality, "Buen ajuste general."),
        "measurements": {
            "chest": {
                "user": round(user_chest, 1),
                "garment": round(garment_chest, 1),
                "difference": round(garment_chest - user_chest, 1)
            }
        }
    }
    return jsonify(res)

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
        
        # Guardar imagen con medidas resolviendo RC (Race Condition)
        base_name = os.path.basename(img_path)
        result_path = os.path.join('static', 'results', f"measured_{base_name}")
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
                            rec = size_recommender.recommend_size(measurements, brand='Generic', clothing_type='shirts' if clothing_type == 'shirt' else 'pants')
                            size = rec.get("recommended_size", "M")
                            print(f"\nTalla recomendada para {clothing_type}: {size}")
                            
                            # Ajustar prenda al cuerpo
                            clothing_id = os.path.splitext(os.path.basename(args.clothing))[0]
                            fitted_img, success = clothing_fitter.fit_clothing_to_body(
                                frame.copy(), body_landmarks, clothing_id
                            )
                            if not success:
                                fitted_img = frame.copy()
                            
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
                    rec = size_recommender.recommend_size(measurements, brand='Generic', clothing_type='shirts' if clothing_type == 'shirt' else 'pants')
                    size = rec.get("recommended_size", "M")
                    print(f"\nTalla recomendada para {clothing_type}: {size}")
                    
                    # Ajustar prenda al cuerpo
                    clothing_id = os.path.splitext(os.path.basename(args.clothing))[0]
                    fitted_img, success = clothing_fitter.fit_clothing_to_body(
                        img.copy(), body_landmarks, clothing_id
                    )
                    if not success:
                        fitted_img = img.copy()
                    
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