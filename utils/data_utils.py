"""
Utilidades para procesamiento y manipulación de datos en el sistema de prueba virtual.
Proporciona funciones para validar, formatear y procesar información relacionada con medidas,
tallas y datos de usuarios.
"""

import json
import os
import re
import csv
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import uuid
import math
import numpy as np

# Configurar logging
logger = logging.getLogger(__name__)


def parse_size_chart(file_path: str) -> Dict[str, Any]:
    """
    Parsea un archivo de tabla de tallas en formato JSON.
    
    Args:
        file_path: Ruta al archivo JSON de tabla de tallas
        
    Returns:
        Diccionario con la información de tallas
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Size chart file not found: {file_path}")
            return {}
            
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        return data
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing size chart: {str(e)}")
        return {}


def validate_measurements(measurements: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Valida que las medidas corporales estén dentro de rangos razonables.
    
    Args:
        measurements: Diccionario con medidas corporales
        
    Returns:
        Tupla (es_válido, lista_de_errores)
    """
    errors = []
    
    # Definir rangos razonables para cada medida (en cm)
    valid_ranges = {
        "height": (120, 220),    # 1.20m - 2.20m
        "weight": (30, 250),     # 30kg - 250kg (en kg)
        "chest": (60, 160),      # 60cm - 160cm
        "waist": (50, 150),      # 50cm - 150cm
        "hip": (70, 170),        # 70cm - 170cm
        "shoulder_width": (30, 60), # 30cm - 60cm
        "arm_length": (40, 90),  # 40cm - 90cm
        "inseam": (60, 110),     # 60cm - 110cm
        "neck": (25, 60),        # 25cm - 60cm
        "thigh": (30, 90)        # 30cm - 90cm
    }
    
    for measure, value in measurements.items():
        if measure in valid_ranges:
            min_val, max_val = valid_ranges[measure]
            
            # Verificar que sea un número y esté dentro del rango
            if not isinstance(value, (int, float)):
                errors.append(f"{measure}: No es un valor numérico")
            elif value < min_val or value > max_val:
                errors.append(f"{measure}: Valor fuera de rango ({min_val}-{max_val})")
    
    return len(errors) == 0, errors


def format_measurements_output(measurements: Dict[str, float], 
                              include_imperial: bool = False,
                              precision: int = 1) -> Dict[str, str]:
    """
    Formatea las medidas para mostrarlas al usuario.
    
    Args:
        measurements: Diccionario con medidas corporales
        include_imperial: Si se debe incluir unidades imperiales
        precision: Número de decimales a mostrar
        
    Returns:
        Diccionario con medidas formateadas
    """
    formatted = {}
    
    # Factores de conversión métrico a imperial
    cm_to_inches = 0.393701
    kg_to_pounds = 2.20462
    
    # Nombres para mostrar
    display_names = {
        "height": "Altura",
        "weight": "Peso",
        "chest": "Contorno de pecho",
        "waist": "Contorno de cintura",
        "hip": "Contorno de cadera",
        "shoulder_width": "Ancho de hombros",
        "arm_length": "Longitud de brazo",
        "inseam": "Entrepierna",
        "neck": "Contorno de cuello",
        "thigh": "Contorno de muslo"
    }
    
    for measure, value in measurements.items():
        if measure in display_names:
            display_name = display_names[measure]
            
            # Formatear con unidades métricas
            if measure == "weight":
                # Peso en kg
                formatted_value = f"{value:.{precision}f} kg"
                if include_imperial:
                    pounds = value * kg_to_pounds
                    formatted_value += f" ({pounds:.{precision}f} lb)"
            elif measure == "height" and value >= 100:
                # Altura en metros y cm
                meters = value / 100
                formatted_value = f"{meters:.{precision}f} m"
                if include_imperial:
                    inches = value * cm_to_inches
                    feet = int(inches / 12)
                    remaining_inches = inches % 12
                    formatted_value += f" ({feet}'{remaining_inches:.{precision}f}\")"
            else:
                # Otras medidas en cm
                formatted_value = f"{value:.{precision}f} cm"
                if include_imperial:
                    inches = value * cm_to_inches
                    formatted_value += f" ({inches:.{precision}f} in)"
                    
            formatted[measure] = {
                "display_name": display_name,
                "value": formatted_value,
                "raw_value": value
            }
    
    return formatted


def calculate_bmi(height_cm: float, weight_kg: float) -> Dict[str, Any]:
    """
    Calcula el Índice de Masa Corporal (IMC).
    
    Args:
        height_cm: Altura en centímetros
        weight_kg: Peso en kilogramos
        
    Returns:
        Diccionario con IMC y categoría
    """
    if height_cm <= 0 or weight_kg <= 0:
        return {
            "bmi": None,
            "category": "Desconocido",
            "in_range": False
        }
    
    # Convertir altura a metros
    height_m = height_cm / 100
    
    # Calcular IMC: peso (kg) / altura² (m)
    bmi = weight_kg / (height_m * height_m)
    
    # Determinar categoría según OMS
    if bmi < 16.0:
        category = "Delgadez severa"
        in_range = False
    elif bmi < 17.0:
        category = "Delgadez moderada"
        in_range = False
    elif bmi < 18.5:
        category = "Delgadez leve"
        in_range = False
    elif bmi < 25.0:
        category = "Peso normal"
        in_range = True
    elif bmi < 30.0:
        category = "Sobrepeso"
        in_range = False
    elif bmi < 35.0:
        category = "Obesidad leve"
        in_range = False
    elif bmi < 40.0:
        category = "Obesidad moderada"
        in_range = False
    else:
        category = "Obesidad mórbida"
        in_range = False
        
    return {
        "bmi": round(bmi, 1),
        "category": category,
        "in_range": in_range
    }


def generate_filename(prefix: str = "image", extension: str = "png") -> str:
    """
    Genera un nombre de archivo único basado en timestamp y UUID.
    
    Args:
        prefix: Prefijo para el nombre del archivo
        extension: Extensión del archivo (sin punto)
        
    Returns:
        Nombre de archivo único
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Tomar los primeros 8 caracteres del UUID
    
    return f"{prefix}_{timestamp}_{unique_id}.{extension}"


def extract_measurements_from_landmarks(landmarks: Dict[str, Dict[str, float]], 
                                       gender: str = "neutral",
                                       height_cm: Optional[float] = None) -> Dict[str, float]:
    """
    Extrae medidas iniciales a partir de landmarks. Esta es una versión simplificada
    y menos precisa que la implementación completa en la clase MeasurementCalculator.
    
    Args:
        landmarks: Diccionario con landmarks detectados
        gender: Género para ajustes específicos
        height_cm: Altura conocida en cm para calibración
        
    Returns:
        Diccionario con medidas estimadas
    """
    # Esta es una implementación muy simplificada
    # La versión completa estaría en core/measurement.py
    measurements = {}
    
    # Estimar un factor de escala basado en la altura si está disponible
    scale_factor = 1.0
    if height_cm and "left_ankle" in landmarks and "nose" in landmarks:
        pixel_height = abs(landmarks["nose"]["y"] - landmarks["left_ankle"]["y"])
        if pixel_height > 0:
            scale_factor = height_cm / (pixel_height * 1.1)  # Factor de ajuste
    
    # Estimar medidas básicas
    if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
        shoulder_width_px = abs(landmarks["left_shoulder"]["x"] - landmarks["right_shoulder"]["x"])
        measurements["shoulder_width"] = shoulder_width_px * scale_factor
    
    if "left_shoulder" in landmarks and "left_hip" in landmarks:
        torso_height_px = abs(landmarks["left_shoulder"]["y"] - landmarks["left_hip"]["y"])
        
        # Estimar contorno de pecho (aproximación muy básica)
        if "right_shoulder" in landmarks:
            chest_width_px = abs(landmarks["left_shoulder"]["x"] - landmarks["right_shoulder"]["x"])
            chest_circumference_px = chest_width_px * 2.5  # Aproximación
            measurements["chest"] = chest_circumference_px * scale_factor
        
        # Estimar cintura (aproximación)
        if "right_hip" in landmarks:
            waist_y = landmarks["left_shoulder"]["y"] + torso_height_px * 0.4
            waist_width_px = abs(landmarks["left_hip"]["x"] - landmarks["right_hip"]["x"]) * 0.9
            waist_circumference_px = waist_width_px * 2.3  # Aproximación
            measurements["waist"] = waist_circumference_px * scale_factor
    
    # Estimar otras medidas
    # (Implementación simplificada)
    
    return measurements


def compare_size_charts(brand1: str, brand2: str, 
                      clothing_type: str = "shirts") -> Dict[str, Dict[str, str]]:
    """
    Compara tablas de tallas entre marcas para crear una tabla de equivalencia.
    
    Args:
        brand1: Primera marca
        brand2: Segunda marca
        clothing_type: Tipo de prenda
        
    Returns:
        Diccionario con equivalencias de tallas
    """
    # Esta función requeriría acceso a las tablas de tallas
    # Por simplicidad, devolvemos un ejemplo
    
    if clothing_type == "shirts":
        return {
            "XS": {"brand1": "XS", "brand2": "S"},
            "S": {"brand1": "S", "brand2": "M"},
            "M": {"brand1": "M", "brand2": "M"},
            "L": {"brand1": "L", "brand2": "L"},
            "XL": {"brand1": "XL", "brand2": "L"}
        }
    elif clothing_type == "pants":
        return {
            "28": {"brand1": "28", "brand2": "29"},
            "30": {"brand1": "30", "brand2": "30-31"},
            "32": {"brand1": "32", "brand2": "32"},
            "34": {"brand1": "34", "brand2": "33-34"},
            "36": {"brand1": "36", "brand2": "35-36"}
        }
    else:
        return {}


def parse_csv_size_chart(file_path: str) -> Dict[str, Any]:
    """
    Parsea un archivo CSV de tabla de tallas.
    
    Args:
        file_path: Ruta al archivo CSV
        
    Returns:
        Diccionario con datos de tallas
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}")
            return {}
        
        result = {
            "sizes": [],
            "measurements": {}
        }
        
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Primera fila como cabeceras
            
            # Extraer nombres de tallas (a partir de la segunda columna)
            for i in range(1, len(headers)):
                result["sizes"].append(headers[i])
            
            # Procesar filas con medidas
            for row in reader:
                if len(row) > 0:
                    measure_name = row[0]
                    values = []
                    
                    for i in range(1, len(row)):
                        if i < len(row) and row[i]:
                            try:
                                values.append(float(row[i]))
                            except ValueError:
                                values.append(row[i])  # Mantener como texto si no es número
                        else:
                            values.append(None)  # Valor vacío
                    
                    result["measurements"][measure_name] = values
        
        return result
    except Exception as e:
        logger.error(f"Error parsing CSV size chart: {str(e)}")
        return {}


def convert_to_standard_size_format(data: Dict[str, Any], 
                                  brand: str, 
                                  clothing_type: str, 
                                  gender: str = "unisex") -> Dict[str, Any]:
    """
    Convierte datos de tallas a un formato estándar para el sistema.
    
    Args:
        data: Datos de tallas en formato crudo
        brand: Nombre de la marca
        clothing_type: Tipo de prenda
        gender: Género
        
    Returns:
        Datos en formato estándar
    """
    try:
        standard_format = {
            "brand": brand,
            "type": clothing_type,
            "gender": gender,
            "sizes": {}
        }
        
        # Verificar estructura básica
        if "sizes" not in data or "measurements" not in data:
            logger.error("Invalid size chart data format")
            return standard_format
        
        sizes = data["sizes"]
        measurements = data["measurements"]
        
        # Procesar cada talla
        for i, size_name in enumerate(sizes):
            size_data = {}
            
            # Procesar cada medida para esta talla
            for measure, values in measurements.items():
                if i < len(values) and values[i] is not None:
                    # Determinar si es un rango o valor único
                    if isinstance(values[i], str) and "-" in values[i]:
                        # Es un rango (ej: "80-85")
                        parts = values[i].split("-")
                        if len(parts) == 2:
                            try:
                                min_val = float(parts[0])
                                max_val = float(parts[1])
                                size_data[f"{measure}_min"] = min_val
                                size_data[f"{measure}_max"] = max_val
                            except ValueError:
                                # Si no se puede convertir, usar como texto
                                size_data[measure] = values[i]
                    else:
                        # Es un valor único
                        if isinstance(values[i], (int, float)):
                            # Crear un pequeño rango alrededor del valor
                            value = float(values[i])
                            size_data[f"{measure}_min"] = value * 0.97  # -3%
                            size_data[f"{measure}_max"] = value * 1.03  # +3%
                        else:
                            # Valor de texto
                            size_data[measure] = values[i]
            
            # Guardar datos de esta talla
            standard_format["sizes"][size_name] = size_data
        
        return standard_format
    except Exception as e:
        logger.error(f"Error converting to standard size format: {str(e)}")
        return {
            "brand": brand,
            "type": clothing_type,
            "gender": gender,
            "sizes": {}
        }


def merge_measurements(previous: Dict[str, float], 
                      current: Dict[str, float], 
                      weight_current: float = 0.7) -> Dict[str, float]:
    """
    Combina medidas anteriores con actuales para mayor estabilidad.
    
    Args:
        previous: Medidas anteriores
        current: Medidas actuales
        weight_current: Peso para las medidas actuales (0-1)
        
    Returns:
        Medidas combinadas
    """
    weight_previous = 1.0 - weight_current
    result = {}
    
    # Usar todas las claves de ambos diccionarios
    all_keys = set(previous.keys()) | set(current.keys())
    
    for key in all_keys:
        if key in previous and key in current:
            # Promedio ponderado si la medida está en ambos conjuntos
            result[key] = previous[key] * weight_previous + current[key] * weight_current
        elif key in current:
            # Solo está en medidas actuales
            result[key] = current[key]
        else:
            # Solo está en medidas anteriores
            result[key] = previous[key]
    
    return result


def validate_email(email: str) -> bool:
    """
    Valida que una dirección de correo tenga formato correcto.
    
    Args:
        email: Dirección de correo a validar
        
    Returns:
        True si el formato es válido
    """
    if not email:
        return False
    
    # Patrón de validación básico para correos
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def anonymize_user_data(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Anonimiza datos de usuario para analytics.
    
    Args:
        user_data: Datos completos del usuario
        
    Returns:
        Datos anonimizados
    """
    if not user_data:
        return {}
    
    anon_data = {}
    
    # Copiar datos básicos no identificables
    if "gender" in user_data:
        anon_data["gender"] = user_data["gender"]
    
    if "height" in user_data:
        anon_data["height"] = user_data["height"]
        
    if "weight" in user_data:
        anon_data["weight"] = user_data["weight"]
    
    # Anonimizar edad si hay fecha de nacimiento
    if "birth_date" in user_data and user_data["birth_date"]:
        try:
            birth_date = datetime.datetime.strptime(user_data["birth_date"], "%Y-%m-%d")
            today = datetime.datetime.now()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            # Agrupar en rangos de edad
            if age < 18:
                age_group = "<18"
            elif age < 25:
                age_group = "18-24"
            elif age < 35:
                age_group = "25-34"
            elif age < 45:
                age_group = "35-44"
            elif age < 55:
                age_group = "45-54"
            elif age < 65:
                age_group = "55-64"
            else:
                age_group = "65+"
                
            anon_data["age_group"] = age_group
        except:
            pass
    
    # Generar un ID anónimo único
    anon_data["anon_id"] = str(uuid.uuid4())
    
    return anon_data


def get_seasonal_recommendations(measurements: Dict[str, float], 
                               season: str = "all") -> List[Dict[str, Any]]:
    """
    Obtiene recomendaciones de prendas basadas en la estación.
    
    Args:
        measurements: Medidas del usuario
        season: Estación (spring, summer, fall, winter, all)
        
    Returns:
        Lista de recomendaciones
    """
    # Esta es una implementación de ejemplo
    recommendations = []
    
    if season in ["summer", "all"]:
        recommendations.append({
            "type": "shirt",
            "style": "T-shirt",
            "description": "Camiseta de manga corta en algodón",
            "season_match": 0.9 if season == "summer" else 0.6
        })
    
    if season in ["winter", "fall", "all"]:
        recommendations.append({
            "type": "shirt",
            "style": "Sweater",
            "description": "Suéter de lana",
            "season_match": 0.9 if season == "winter" else 0.7
        })
    
    # Se añadirían más recomendaciones basadas en medidas y preferencias
    
    return recommendations