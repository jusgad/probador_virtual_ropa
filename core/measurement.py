"""
Módulo para el cálculo de medidas corporales a partir de landmarks.
Implementa algoritmos para determinar medidas como contorno de pecho,
cintura, cadera, longitud de brazos y piernas, etc.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import json
import os


class MeasurementCalculator:
    """
    Clase para calcular medidas corporales basadas en landmarks.
    Utiliza fórmulas antropométricas y factores de escala para
    obtener medidas precisas en centímetros.
    """

    def __init__(self, calibration_file: str = 'data/references/calibration.json'):
        """
        Inicializa el calculador de medidas.

        Args:
            calibration_file: Ruta al archivo de calibración con factores de conversión
        """
        self.calibration_data = self._load_calibration(calibration_file)
        self.reference_height_cm = 170.0  # Altura de referencia en cm
        self.pixel_to_cm_ratio = 1.0  # Ratio de conversión inicial
        
    def _load_calibration(self, calibration_file: str) -> Dict:
        """
        Carga datos de calibración desde un archivo JSON.
        
        Args:
            calibration_file: Ruta al archivo de calibración
            
        Returns:
            Diccionario con datos de calibración
        """
        try:
            if os.path.exists(calibration_file):
                with open(calibration_file, 'r') as f:
                    return json.load(f)
            else:
                # Valores predeterminados si no existe el archivo
                return {
                    "body_ratios": {
                        "shoulder_to_height": 0.259,
                        "chest_to_height": 0.275,
                        "waist_to_height": 0.44,
                        "hip_to_height": 0.53,
                        "inseam_to_height": 0.47
                    },
                    "adjustment_factors": {
                        "chest": 1.15,
                        "waist": 1.08,
                        "hip": 1.12,
                        "shoulder": 1.05,
                        "arm_length": 1.02,
                        "leg_length": 1.03
                    },
                    "gender_differences": {
                        "male": {
                            "chest": 1.05,
                            "waist": 0.98,
                            "hip": 0.95
                        },
                        "female": {
                            "chest": 1.02,
                            "waist": 1.03,
                            "hip": 1.08
                        }
                    }
                }
        except Exception as e:
            print(f"Error al cargar datos de calibración: {e}")
            return {}

    def calibrate_with_height(self, landmarks: Dict[str, Dict[str, float]], known_height_cm: float) -> None:
        """
        Calibra las medidas usando una altura conocida.
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            known_height_cm: Altura conocida del sujeto en centímetros
        """
        if "left_ankle" not in landmarks or "left_shoulder" not in landmarks:
            return
            
        # Calcular altura en píxeles (de hombro a tobillo)
        shoulder_y = landmarks["left_shoulder"]["y"]
        ankle_y = landmarks["left_ankle"]["y"]
        height_px = abs(ankle_y - shoulder_y)
        
        # Ajustar por la proporción de altura total (hombro-tobillo no es la altura completa)
        estimated_full_height_px = height_px / 0.85  # El 85% de la altura suele ser hombro-tobillo
        
        # Establecer ratio de píxel a cm
        self.pixel_to_cm_ratio = known_height_cm / estimated_full_height_px
        self.reference_height_cm = known_height_cm

    def calculate_all_measurements(
        self, 
        landmarks: Dict[str, Dict[str, float]], 
        gender: str = "neutral", 
        height_cm: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calcula todas las medidas corporales relevantes.
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            gender: Género para ajustes específicos ("male", "female", o "neutral")
            height_cm: Altura en cm si se conoce, para calibración
            
        Returns:
            Diccionario con todas las medidas calculadas en centímetros
        """
        # Calibrar con altura si está disponible
        if height_cm:
            self.calibrate_with_height(landmarks, height_cm)
            
        # Calcular medidas básicas
        measurements = {
            "height": self._calculate_height(landmarks),
            "shoulder_width": self._calculate_shoulder_width(landmarks, gender),
            "chest": self._calculate_chest(landmarks, gender),
            "waist": self._calculate_waist(landmarks, gender),
            "hip": self._calculate_hip(landmarks, gender),
            "arm_length": self._calculate_arm_length(landmarks, gender),
            "inseam": self._calculate_inseam(landmarks, gender),
            "neck": self._calculate_neck(landmarks, gender),
            "thigh": self._calculate_thigh(landmarks, gender)
        }
        
        # Calcular medidas derivadas
        measurements["bust"] = measurements["chest"]  # Para mujeres, similar al pecho
        
        # Aplicar factores de ajuste final según género
        gender_factors = self.calibration_data.get("gender_differences", {}).get(gender, {})
        for measure, factor in gender_factors.items():
            if measure in measurements:
                measurements[measure] *= factor
                
        return measurements

    def _distance(self, point1: Dict[str, float], point2: Dict[str, float]) -> float:
        """
        Calcula la distancia euclidiana entre dos puntos.
        
        Args:
            point1: Primer punto con claves 'x' e 'y'
            point2: Segundo punto con claves 'x' e 'y'
            
        Returns:
            Distancia en píxeles
        """
        return math.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2)
    
    def _pixels_to_cm(self, pixels: float) -> float:
        """
        Convierte píxeles a centímetros usando el ratio de calibración.
        
        Args:
            pixels: Distancia en píxeles
            
        Returns:
            Distancia en centímetros
        """
        return pixels * self.pixel_to_cm_ratio
        
    def _calculate_height(self, landmarks: Dict[str, Dict[str, float]]) -> float:
        """
        Calcula la altura del cuerpo.
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            
        Returns:
            Altura en centímetros
        """
        # Si tenemos landmarks para los pies y la cabeza
        if "left_ankle" in landmarks and "nose" in landmarks:
            height_px = abs(landmarks["nose"]["y"] - landmarks["left_ankle"]["y"])
            # Ajustar porque el tobillo no es el punto más bajo y la nariz no es el más alto
            height_px = height_px * 1.15  # Factor de ajuste
            return self._pixels_to_cm(height_px)
        else:
            # Si faltan landmarks, usar un valor predeterminado
            return self.reference_height_cm
            
    def _calculate_shoulder_width(self, landmarks: Dict[str, Dict[str, float]], gender: str) -> float:
        """
        Calcula el ancho de hombros.
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            gender: Género para ajustes específicos
            
        Returns:
            Ancho de hombros en centímetros
        """
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
            width_px = self._distance(landmarks["left_shoulder"], landmarks["right_shoulder"])
            width_cm = self._pixels_to_cm(width_px)
            
            # Aplicar factor de ajuste
            adjustment = self.calibration_data.get("adjustment_factors", {}).get("shoulder", 1.05)
            return width_cm * adjustment
        else:
            # Estimación basada en altura si no hay landmarks
            ratio = self.calibration_data.get("body_ratios", {}).get("shoulder_to_height", 0.259)
            return self.reference_height_cm * ratio
            
    def _calculate_chest(self, landmarks: Dict[str, Dict[str, float]], gender: str) -> float:
        """
        Calcula el contorno de pecho.
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            gender: Género para ajustes específicos
            
        Returns:
            Contorno de pecho en centímetros
        """
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
            # Estimar la posición de los puntos del pecho
            chest_width_px = self._distance(landmarks["left_shoulder"], landmarks["right_shoulder"])
            
            # El perímetro del pecho se estima como una elipse
            chest_depth_px = chest_width_px * 0.7  # Profundidad estimada del torso
            
            # Perímetro aproximado de una elipse
            a = chest_width_px / 2
            b = chest_depth_px / 2
            perimeter_px = 2 * math.pi * math.sqrt((a*a + b*b) / 2)
            
            chest_cm = self._pixels_to_cm(perimeter_px)
            
            # Aplicar factor de ajuste
            adjustment = self.calibration_data.get("adjustment_factors", {}).get("chest", 1.15)
            return chest_cm * adjustment
        else:
            # Estimación basada en altura
            ratio = self.calibration_data.get("body_ratios", {}).get("chest_to_height", 0.275)
            chest_to_height = self.reference_height_cm * ratio
            
            # Convertir de proporción a circunferencia
            return chest_to_height * 2 * math.pi
            
    def _calculate_waist(self, landmarks: Dict[str, Dict[str, float]], gender: str) -> float:
        """
        Calcula el contorno de cintura.
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            gender: Género para ajustes específicos
            
        Returns:
            Contorno de cintura en centímetros
        """
        if "left_hip" in landmarks and "right_hip" in landmarks:
            # Estimar la posición de la cintura (punto medio entre hombros y caderas)
            left_shoulder_y = landmarks["left_shoulder"]["y"]
            left_hip_y = landmarks["left_hip"]["y"]
            waist_y = left_shoulder_y + (left_hip_y - left_shoulder_y) * 0.4
            
            # Estimar el ancho de la cintura
            hip_width_px = self._distance(landmarks["left_hip"], landmarks["right_hip"])
            waist_width_px = hip_width_px * 0.9  # La cintura suele ser un 90% del ancho de cadera
            
            # El perímetro de la cintura se estima como una elipse
            waist_depth_px = waist_width_px * 0.7  # Profundidad estimada
            
            # Perímetro aproximado de una elipse
            a = waist_width_px / 2
            b = waist_depth_px / 2
            perimeter_px = 2 * math.pi * math.sqrt((a*a + b*b) / 2)
            
            waist_cm = self._pixels_to_cm(perimeter_px)
            
            # Aplicar factor de ajuste
            adjustment = self.calibration_data.get("adjustment_factors", {}).get("waist", 1.08)
            return waist_cm * adjustment
        else:
            # Estimación basada en altura
            ratio = self.calibration_data.get("body_ratios", {}).get("waist_to_height", 0.44)
            waist_to_height = self.reference_height_cm * ratio
            
            # Convertir de proporción a circunferencia
            return waist_to_height * 2 * math.pi * 0.8  # Factor de ajuste
            
    def _calculate_hip(self, landmarks: Dict[str, Dict[str, float]], gender: str) -> float:
        """
        Calcula el contorno de cadera.
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            gender: Género para ajustes específicos
            
        Returns:
            Contorno de cadera en centímetros
        """
        if "left_hip" in landmarks and "right_hip" in landmarks:
            # Medir el ancho de cadera
            hip_width_px = self._distance(landmarks["left_hip"], landmarks["right_hip"])
            
            # El perímetro de la cadera se estima como una elipse
            hip_depth_px = hip_width_px * 0.8  # Profundidad estimada
            
            # Perímetro aproximado de una elipse
            a = hip_width_px / 2
            b = hip_depth_px / 2
            perimeter_px = 2 * math.pi * math.sqrt((a*a + b*b) / 2)
            
            hip_cm = self._pixels_to_cm(perimeter_px)
            
            # Aplicar factor de ajuste
            adjustment = self.calibration_data.get("adjustment_factors", {}).get("hip", 1.12)
            return hip_cm * adjustment
        else:
            # Estimación basada en altura
            ratio = self.calibration_data.get("body_ratios", {}).get("hip_to_height", 0.53)
            hip_to_height = self.reference_height_cm * ratio
            
            # Convertir de proporción a circunferencia
            return hip_to_height * 2 * math.pi * 0.95  # Factor de ajuste
    
    def _calculate_arm_length(self, landmarks: Dict[str, Dict[str, float]], gender: str) -> float:
        """
        Calcula la longitud de brazo.
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            gender: Género para ajustes específicos
            
        Returns:
            Longitud de brazo en centímetros
        """
        if "left_shoulder" in landmarks and "left_elbow" in landmarks and "left_wrist" in landmarks:
            # Suma de las distancias hombro-codo y codo-muñeca
            upper_arm_px = self._distance(landmarks["left_shoulder"], landmarks["left_elbow"])
            forearm_px = self._distance(landmarks["left_elbow"], landmarks["left_wrist"])
            total_length_px = upper_arm_px + forearm_px
            
            arm_length_cm = self._pixels_to_cm(total_length_px)
            
            # Aplicar factor de ajuste
            adjustment = self.calibration_data.get("adjustment_factors", {}).get("arm_length", 1.02)
            return arm_length_cm * adjustment
        else:
            # Estimación basada en altura
            return self.reference_height_cm * 0.33  # ~33% de la altura
            
    def _calculate_inseam(self, landmarks: Dict[str, Dict[str, float]], gender: str) -> float:
        """
        Calcula la longitud de entrepierna (inseam).
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            gender: Género para ajustes específicos
            
        Returns:
            Longitud de entrepierna en centímetros
        """
        if "left_hip" in landmarks and "left_knee" in landmarks and "left_ankle" in landmarks:
            # Distancia desde la cadera hasta el tobillo
            hip_to_knee_px = self._distance(landmarks["left_hip"], landmarks["left_knee"])
            knee_to_ankle_px = self._distance(landmarks["left_knee"], landmarks["left_ankle"])
            
            # Sumamos ambas partes
            inseam_px = hip_to_knee_px + knee_to_ankle_px
            
            inseam_cm = self._pixels_to_cm(inseam_px)
            
            # Aplicar factor de ajuste
            adjustment = self.calibration_data.get("adjustment_factors", {}).get("leg_length", 1.03)
            return inseam_cm * adjustment
        else:
            # Estimación basada en altura
            ratio = self.calibration_data.get("body_ratios", {}).get("inseam_to_height", 0.47)
            return self.reference_height_cm * ratio
            
    def _calculate_neck(self, landmarks: Dict[str, Dict[str, float]], gender: str) -> float:
        """
        Calcula el contorno de cuello.
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            gender: Género para ajustes específicos
            
        Returns:
            Contorno de cuello en centímetros
        """
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks and "nose" in landmarks:
            # Estimar el contorno del cuello basado en la distancia entre hombros
            shoulder_width_px = self._distance(landmarks["left_shoulder"], landmarks["right_shoulder"])
            neck_width_px = shoulder_width_px * 0.3  # Aproximación
            
            # Estimar el perímetro como un círculo
            neck_perimeter_px = neck_width_px * math.pi
            
            neck_cm = self._pixels_to_cm(neck_perimeter_px)
            return neck_cm
        else:
            # Estimación basada en altura
            return self.reference_height_cm * 0.2
            
    def _calculate_thigh(self, landmarks: Dict[str, Dict[str, float]], gender: str) -> float:
        """
        Calcula el contorno de muslo.
        
        Args:
            landmarks: Diccionario con los landmarks detectados
            gender: Género para ajustes específicos
            
        Returns:
            Contorno de muslo en centímetros
        """
        if "left_hip" in landmarks and "left_knee" in landmarks:
            # Estimar el ancho del muslo (a un tercio de distancia entre cadera y rodilla)
            hip_to_knee_vector = {
                "x": landmarks["left_knee"]["x"] - landmarks["left_hip"]["x"],
                "y": landmarks["left_knee"]["y"] - landmarks["left_hip"]["y"]
            }
            
            thigh_point = {
                "x": landmarks["left_hip"]["x"] + hip_to_knee_vector["x"] * 0.33,
                "y": landmarks["left_hip"]["y"] + hip_to_knee_vector["y"] * 0.33
            }
            
            # Estimar el ancho del muslo basado en la distancia de las caderas
            if "right_hip" in landmarks:
                hip_width_px = self._distance(landmarks["left_hip"], landmarks["right_hip"])
                thigh_width_px = hip_width_px * 0.5  # Aproximación
                
                # Estimar el perímetro como una elipse
                thigh_depth_px = thigh_width_px * 0.9  # Profundidad estimada
                
                # Perímetro aproximado de una elipse
                a = thigh_width_px / 2
                b = thigh_depth_px / 2
                perimeter_px = 2 * math.pi * math.sqrt((a*a + b*b) / 2)
                
                thigh_cm = self._pixels_to_cm(perimeter_px)
                return thigh_cm
            else:
                return self.reference_height_cm * 0.28
        else:
            # Estimación basada en altura
            return self.reference_height_cm * 0.28
            
    def estimate_clothing_size(self, measurements: Dict[str, float], gender: str, size_chart: Dict) -> Dict[str, str]:
        """
        Estima las tallas de ropa basadas en las medidas calculadas.
        
        Args:
            measurements: Diccionario con medidas calculadas
            gender: Género ("male", "female", o "neutral")
            size_chart: Tabla de tallas de referencia
            
        Returns:
            Diccionario con tallas estimadas para diferentes prendas
        """
        sizes = {}
        
        # Determinar talla de camisa/blusa
        if "chest" in measurements and "shoulder_width" in measurements:
            chest = measurements["chest"]
            shoulder = measurements["shoulder_width"]
            
            # Buscar en la tabla de tallas correspondiente
            shirt_sizes = size_chart.get("shirts", {}).get(gender, {})
            for size_name, size_range in shirt_sizes.items():
                if (size_range["chest_min"] <= chest <= size_range["chest_max"] and
                    size_range["shoulder_min"] <= shoulder <= size_range["shoulder_max"]):
                    sizes["shirt"] = size_name
                    break
            
            # Si no se encuentra una coincidencia exacta, elegir la más cercana
            if "shirt" not in sizes and shirt_sizes:
                closest_size = None
                min_diff = float('inf')
                
                for size_name, size_range in shirt_sizes.items():
                    chest_mid = (size_range["chest_min"] + size_range["chest_max"]) / 2
                    shoulder_mid = (size_range["shoulder_min"] + size_range["shoulder_max"]) / 2
                    
                    diff = abs(chest - chest_mid) + abs(shoulder - shoulder_mid)
                    if diff < min_diff:
                        min_diff = diff
                        closest_size = size_name
                
                if closest_size:
                    sizes["shirt"] = closest_size
                    
        # Determinar talla de pantalones
        if "waist" in measurements and "hip" in measurements and "inseam" in measurements:
            waist = measurements["waist"]
            hip = measurements["hip"]
            inseam = measurements["inseam"]
            
            # Buscar en la tabla de tallas correspondiente
            pant_sizes = size_chart.get("pants", {}).get(gender, {})
            for size_name, size_range in pant_sizes.items():
                if (size_range["waist_min"] <= waist <= size_range["waist_max"] and
                    size_range["hip_min"] <= hip <= size_range["hip_max"]):
                    sizes["pants"] = size_name
                    break
            
            # Si no se encuentra una coincidencia exacta, elegir la más cercana
            if "pants" not in sizes and pant_sizes:
                closest_size = None
                min_diff = float('inf')
                
                for size_name, size_range in pant_sizes.items():
                    waist_mid = (size_range["waist_min"] + size_range["waist_max"]) / 2
                    hip_mid = (size_range["hip_min"] + size_range["hip_max"]) / 2
                    
                    diff = abs(waist - waist_mid) + abs(hip - hip_mid)
                    if diff < min_diff:
                        min_diff = diff
                        closest_size = size_name
                
                if closest_size:
                    sizes["pants"] = closest_size
                    
        return sizes