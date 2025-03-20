"""
Módulo para ajustar prendas de ropa a cuerpos detectados.
Implementa técnicas de deformación de imágenes y superposición
para simular cómo quedaría una prenda en un cuerpo específico.
"""

import cv2
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import os
from utils.image_utils import overlay_images, resize_image_to_height, warp_image


class ClothingFitter:
    """
    Clase para ajustar virtualmente prendas de ropa al cuerpo.
    Utiliza técnicas de deformación de imágenes y transformación
    para adaptar prendas a diferentes cuerpos.
    """

    def __init__(self, clothes_directory: str = 'data/clothes'):
        """
        Inicializa el ajustador de ropa.

        Args:
            clothes_directory: Directorio donde se almacenan las imágenes de ropa
        """
        self.clothes_directory = clothes_directory
        self.clothing_templates = self._load_clothing_templates()
        
    def _load_clothing_templates(self) -> Dict[str, Dict[str, any]]:
        """
        Carga las plantillas de ropa desde el directorio.
        
        Returns:
            Diccionario con información de las plantillas de ropa
        """
        templates = {}
        
        # Cargamos camisas/blusas
        shirt_directory = os.path.join(self.clothes_directory, 'shirts')
        if os.path.exists(shirt_directory):
            templates["shirts"] = {}
            for filename in os.listdir(shirt_directory):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(shirt_directory, filename)
                    
                    # Extraer información del nombre del archivo (ejemplo: shirt_blue_M.png)
                    name_parts = os.path.splitext(filename)[0].split('_')
                    
                    if len(name_parts) >= 3:
                        color = name_parts[1]
                        size = name_parts[2]
                        shirt_id = f"shirt_{color}_{size}"
                        
                        # Cargar imagen con canal alfa (transparencia)
                        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                        
                        # Si la imagen no tiene canal alfa, añadirlo
                        if image.shape[2] == 3:
                            b, g, r = cv2.split(image)
                            alpha = np.ones(b.shape, dtype=b.dtype) * 255
                            image = cv2.merge((b, g, r, alpha))
                            
                        # Detectar puntos clave de la prenda (simplificado)
                        key_points = self._detect_clothing_keypoints(image, "shirt")
                        
                        templates["shirts"][shirt_id] = {
                            "image": image,
                            "size": size,
                            "color": color,
                            "type": "shirt",
                            "key_points": key_points
                        }
                        
        # Cargamos pantalones
        pants_directory = os.path.join(self.clothes_directory, 'pants')
        if os.path.exists(pants_directory):
            templates["pants"] = {}
            for filename in os.listdir(pants_directory):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(pants_directory, filename)
                    
                    # Extraer información del nombre
                    name_parts = os.path.splitext(filename)[0].split('_')
                    
                    if len(name_parts) >= 3:
                        color = name_parts[1]
                        size = name_parts[2]
                        pants_id = f"pants_{color}_{size}"
                        
                        # Cargar imagen con canal alfa
                        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                        
                        # Si la imagen no tiene canal alfa, añadirlo
                        if image.shape[2] == 3:
                            b, g, r = cv2.split(image)
                            alpha = np.ones(b.shape, dtype=b.dtype) * 255
                            image = cv2.merge((b, g, r, alpha))
                            
                        # Detectar puntos clave
                        key_points = self._detect_clothing_keypoints(image, "pants")
                        
                        templates["pants"][pants_id] = {
                            "image": image,
                            "size": size,
                            "color": color,
                            "type": "pants",
                            "key_points": key_points
                        }
                        
        return templates
        
    def _detect_clothing_keypoints(self, image: np.ndarray, clothing_type: str) -> Dict[str, Tuple[int, int]]:
        """
        Detecta puntos clave en una prenda de ropa.
        
        Args:
            image: Imagen de la prenda
            clothing_type: Tipo de prenda ("shirt" o "pants")
            
        Returns:
            Diccionario con puntos clave
        """
        # En una implementación real, esto usaría técnicas como detección de contornos,
        # segmentación o modelos pre-entrenados. Aquí usamos una aproximación simplificada.
        
        height, width = image.shape[:2]
        key_points = {}
        
        if clothing_type == "shirt":
            # Puntos clave para camisas/blusas
            key_points = {
                "collar": (width // 2, int(height * 0.05)),
                "left_shoulder": (int(width * 0.2), int(height * 0.1)),
                "right_shoulder": (int(width * 0.8), int(height * 0.1)),
                "left_sleeve": (int(width * 0.05), int(height * 0.25)),
                "right_sleeve": (int(width * 0.95), int(height * 0.25)),
                "left_armpit": (int(width * 0.2), int(height * 0.3)),
                "right_armpit": (int(width * 0.8), int(height * 0.3)),
                "left_waist": (int(width * 0.2), int(height * 0.9)),
                "right_waist": (int(width * 0.8), int(height * 0.9)),
                "chest_center": (width // 2, int(height * 0.3)),
                "waist_center": (width // 2, int(height * 0.9))
            }
        elif clothing_type == "pants":
            # Puntos clave para pantalones
            key_points = {
                "waist_center": (width // 2, int(height * 0.05)),
                "left_waist": (int(width * 0.3), int(height * 0.05)),
                "right_waist": (int(width * 0.7), int(height * 0.05)),
                "crotch": (width // 2, int(height * 0.3)),
                "left_hip": (int(width * 0.3), int(height * 0.2)),
                "right_hip": (int(width * 0.7), int(height * 0.2)),
                "left_knee": (int(width * 0.35), int(height * 0.6)),
                "right_knee": (int(width * 0.65), int(height * 0.6)),
                "left_ankle": (int(width * 0.35), int(height * 0.95)),
                "right_ankle": (int(width * 0.65), int(height * 0.95))
            }
        
        return key_points
        
    def get_clothing_by_id(self, clothing_id: str) -> Optional[Dict[str, any]]:
        """
        Obtiene una prenda por su ID.
        
        Args:
            clothing_id: ID de la prenda
            
        Returns:
            Diccionario con datos de la prenda o None si no existe
        """
        # Ejemplo: "shirt_blue_M"
        if "_" not in clothing_id:
            return None
            
        parts = clothing_id.split("_")
        if len(parts) < 3:
            return None
            
        clothing_type = parts[0] + "s"  # "shirt" -> "shirts"
        
        if clothing_type in self.clothing_templates:
            return self.clothing_templates[clothing_type].get(clothing_id)
        
        return None
        
    def fit_clothing_to_body(
        self, 
        image: np.ndarray, 
        landmarks: Dict[str, Dict[str, float]], 
        clothing_id: str
    ) -> Tuple[np.ndarray, bool]:
        """
        Ajusta una prenda de ropa al cuerpo en la imagen.
        
        Args:
            image: Imagen del cuerpo
            landmarks: Landmarks del cuerpo
            clothing_id: ID de la prenda a ajustar
            
        Returns:
            Tupla con (imagen resultante, éxito)
        """
        clothing = self.get_clothing_by_id(clothing_id)
        if not clothing:
            return image, False
            
        clothing_type = clothing["type"]
        
        if clothing_type == "shirt":
            return self._fit_shirt(image, landmarks, clothing)
        elif clothing_type == "pants":
            return self._fit_pants(image, landmarks, clothing)
        else:
            return image, False
    
    def _fit_shirt(
        self, 
        image: np.ndarray, 
        landmarks: Dict[str, Dict[str, float]], 
        shirt_data: Dict[str, any]
    ) -> Tuple[np.ndarray, bool]:
        """
        Ajusta una camisa/blusa al cuerpo.
        
        Args:
            image: Imagen del cuerpo
            landmarks: Landmarks del cuerpo
            shirt_data: Datos de la camisa
            
        Returns:
            Tupla con (imagen resultante, éxito)
        """
        # Verificar que tenemos los landmarks necesarios
        required_landmarks = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not all(lm in landmarks for lm in required_landmarks):
            return image, False
            
        # Obtener dimensiones corporales
        shoulder_width = abs(landmarks["left_shoulder"]["x"] - landmarks["right_shoulder"]["x"])
        torso_height = abs(landmarks["left_shoulder"]["y"] - landmarks["left_hip"]["y"])
        
        # Obtener imagen de la camisa
        shirt_img = shirt_data["image"].copy()
        shirt_key_points = shirt_data["key_points"]
        
        # Calcular escala para ajustar al ancho de hombros
        original_width = abs(shirt_key_points["left_shoulder"][0] - shirt_key_points["right_shoulder"][0])
        scale_factor = shoulder_width / original_width
        
        # Escalar la camisa manteniendo la proporción
        new_width = int(shirt_img.shape[1] * scale_factor)
        new_height = int(shirt_img.shape[0] * scale_factor)
        shirt_resized = cv2.resize(shirt_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Actualizar puntos clave después del escalado
        scaled_key_points = {}
        for point_name, (x, y) in shirt_key_points.items():
            scaled_key_points[point_name] = (int(x * scale_factor), int(y * scale_factor))
        
        # Crear puntos de origen y destino para la transformación
        src_points = np.array([
            scaled_key_points["left_shoulder"],
            scaled_key_points["right_shoulder"],
            scaled_key_points["left_waist"],
            scaled_key_points["right_waist"]
        ], dtype=np.float32)
        
        dst_points = np.array([
            (int(landmarks["left_shoulder"]["x"]), int(landmarks["left_shoulder"]["y"])),
            (int(landmarks["right_shoulder"]["x"]), int(landmarks["right_shoulder"]["y"])),
            (int(landmarks["left_hip"]["x"]), int(landmarks["left_hip"]["y"] - torso_height * 0.1)),  # Ajuste para la cintura
            (int(landmarks["right_hip"]["x"]), int(landmarks["right_hip"]["y"] - torso_height * 0.1))
        ], dtype=np.float32)
        
        # Calcular matriz de transformación
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Aplicar transformación de perspectiva
        warped_shirt = cv2.warpPerspective(
            shirt_resized, 
            transform_matrix, 
            (image.shape[1], image.shape[0]), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0, 0)
        )
        
        # Superponer la camisa en la imagen original
        result_image = overlay_images(image, warped_shirt)
        
        return result_image, True
        
    def _fit_pants(
        self, 
        image: np.ndarray, 
        landmarks: Dict[str, Dict[str, float]], 
        pants_data: Dict[str, any]
    ) -> Tuple[np.ndarray, bool]:
        """
        Ajusta unos pantalones al cuerpo.
        
        Args:
            image: Imagen del cuerpo
            landmarks: Landmarks del cuerpo
            pants_data: Datos de los pantalones
            
        Returns:
            Tupla con (imagen resultante, éxito)
        """
        # Verificar que tenemos los landmarks necesarios
        required_landmarks = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        if not all(lm in landmarks for lm in required_landmarks):
            return image, False
            
        # Obtener dimensiones corporales
        hip_width = abs(landmarks["left_hip"]["x"] - landmarks["right_hip"]["x"])
        leg_length = abs(landmarks["left_hip"]["y"] - landmarks["left_ankle"]["y"])
        
        # Obtener imagen de los pantalones
        pants_img = pants_data["image"].copy()
        pants_key_points = pants_data["key_points"]
        
        # Calcular escala para ajustar al ancho de cadera
        original_width = abs(pants_key_points["left_waist"][0] - pants_key_points["right_waist"][0])
        scale_factor = hip_width / original_width
        
        # Escalar los pantalones manteniendo la proporción
        new_width = int(pants_img.shape[1] * scale_factor)
        new_height = int(pants_img.shape[0] * scale_factor)
        pants_resized = cv2.resize(pants_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Actualizar puntos clave después del escalado
        scaled_key_points = {}
        for point_name, (x, y) in pants_key_points.items():
            scaled_key_points[point_name] = (int(x * scale_factor), int(y * scale_factor))
        
        # Crear puntos de origen y destino para la transformación
        src_points = np.array([
            scaled_key_points["left_waist"],
            scaled_key_points["right_waist"],
            scaled_key_points["left_ankle"],
            scaled_key_points["right_ankle"]
        ], dtype=np.float32)
        
        dst_points = np.array([
            (int(landmarks["left_hip"]["x"]), int(landmarks["left_hip"]["y"])),
            (int(landmarks["right_hip"]["x"]), int(landmarks["right_hip"]["y"])),
            (int(landmarks["left_ankle"]["x"]), int(landmarks["left_ankle"]["y"])),
            (int(landmarks["right_ankle"]["x"]), int(landmarks["right_ankle"]["y"]))
        ], dtype=np.float32)
        
        # Calcular matriz de transformación
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Aplicar transformación de perspectiva
        warped_pants = cv2.warpPerspective(
            pants_resized, 
            transform_matrix, 
            (image.shape[1], image.shape[0]), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0, 0)
        )
        
        # Superponer los pantalones en la imagen original
        result_image = overlay_images(image, warped_pants)
        
        return result_image, True
    
    def create_ensemble(
        self, 
        image: np.ndarray, 
        landmarks: Dict[str, Dict[str, float]], 
        clothing_ids: List[str]
    ) -> np.ndarray:
        """
        Crea un conjunto de varias prendas ajustadas al cuerpo.
        
        Args:
            image: Imagen del cuerpo
            landmarks: Landmarks del cuerpo
            clothing_ids: Lista de IDs de prendas a ajustar
            
        Returns:
            Imagen resultante con todas las prendas
        """
        result = image.copy()
        success_count = 0
        
        # Ordenar prendas para que las capas inferiores se apliquen primero
        def get_clothing_layer(clothing_id):
            clothing_type = clothing_id.split("_")[0]
            if clothing_type == "pants":
                return 1  # Pantalones primero
            elif clothing_type == "shirt":
                return 2  # Luego camisas
            return 10  # Otros accesorios después
            
        sorted_clothing_ids = sorted(clothing_ids, key=get_clothing_layer)
        
        # Aplicar cada prenda en orden
        for clothing_id in sorted_clothing_ids:
            result, success = self.fit_clothing_to_body(result, landmarks, clothing_id)
            if success:
                success_count += 1
                
        return result

    def get_available_clothes(self, clothing_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Obtiene información sobre la ropa disponible.
        
        Args:
            clothing_type: Tipo de ropa (opcional) para filtrar
            
        Returns:
            Diccionario con información de ropa disponible
        """
        available = {}
        
        if clothing_type and clothing_type + "s" in self.clothing_templates:
            category = clothing_type + "s"
            available[category] = list(self.clothing_templates[category].keys())
        elif not clothing_type:
            for category, items in self.clothing_templates.items():
                available[category] = list(items.keys())
                
        return available
        
    def __del__(self):
        """Método para liberar recursos al destruir la instancia"""
        # Podríamos liberar memoria, cerrar archivos, etc.
        pass