"""
Módulo para la detección de cuerpos y puntos de referencia anatómicos.
Utiliza modelos de visión por computadora para identificar y localizar
las partes clave del cuerpo humano en imágenes.
"""

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from typing import Dict, List, Tuple, Optional

from utils.image_utils import preprocess_image, resize_image


class BodyDetector:
    """
    Clase para detectar el cuerpo humano y sus puntos clave en imágenes.
    Utiliza MediaPipe o modelos similares para la detección de pose.
    """

    def __init__(self, model_path: str = 'models/pose/pose_model', use_gpu: bool = False):
        """
        Inicializa el detector de cuerpo.

        Args:
            model_path: Ruta al modelo pre-entrenado
            use_gpu: Si se debe utilizar aceleración GPU
        """
        self.use_gpu = use_gpu
        
        # Configuración para utilizar GPU o CPU
        if self.use_gpu:
            self.device = tf.device('/GPU:0')
        else:
            self.device = tf.device('/CPU:0')

        # Inicializar MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        # Opcionalmente, cargar un modelo personalizado
        try:
            self.custom_model = tf.saved_model.load(model_path)
        except Exception as e:
            print(f"No se pudo cargar el modelo personalizado: {e}")
            self.custom_model = None
            
        self.landmark_names = self._get_landmark_names()

    def _get_landmark_names(self) -> Dict[int, str]:
        """
        Devuelve un diccionario con los nombres de los puntos de referencia.
        
        Returns:
            Diccionario que mapea índices a nombres de landmarks
        """
        return {
            0: "nose",
            1: "left_eye_inner",
            2: "left_eye",
            3: "left_eye_outer",
            4: "right_eye_inner",
            5: "right_eye",
            6: "right_eye_outer",
            7: "left_ear",
            8: "right_ear",
            9: "mouth_left",
            10: "mouth_right",
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
            17: "left_pinky",
            18: "right_pinky",
            19: "left_index",
            20: "right_index",
            21: "left_thumb",
            22: "right_thumb",
            23: "left_hip",
            24: "right_hip",
            25: "left_knee",
            26: "right_knee",
            27: "left_ankle",
            28: "right_ankle",
            29: "left_heel",
            30: "right_heel",
            31: "left_foot_index",
            32: "right_foot_index"
        }

    def detect(self, image: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Detecta los puntos de referencia del cuerpo en la imagen.

        Args:
            image: Imagen como array numpy en formato BGR

        Returns:
            Diccionario con puntos de referencia detectados y sus coordenadas
        """
        # Preprocesar la imagen
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        # Detectar puntos de referencia usando MediaPipe
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return {}
        
        # Convertir los resultados a un diccionario más manejable
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in self.landmark_names:
                name = self.landmark_names[idx]
                landmarks[name] = {
                    'x': landmark.x * width,  # Convertir a coordenadas de píxeles
                    'y': landmark.y * height,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
        
        return landmarks
    
    def detect_with_segmentation(self, image: np.ndarray) -> Tuple[Dict[str, Dict[str, float]], np.ndarray]:
        """
        Detecta puntos de referencia y además devuelve una máscara de segmentación del cuerpo.

        Args:
            image: Imagen como array numpy

        Returns:
            Tupla con (puntos de referencia, máscara de segmentación)
        """
        # Primero obtener los landmarks
        landmarks = self.detect(image)
        
        # Luego generar una máscara de segmentación (simplificada aquí)
        # En un sistema real, esto podría usar otro modelo específico para segmentación
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Usar un modelo de segmentación o técnicas basadas en landmarks para crear la máscara
        if landmarks and self.custom_model:
            with self.device:
                # Aquí iría la lógica para generar la máscara con el modelo personalizado
                # Simplificado para este ejemplo
                input_tensor = preprocess_image(image)
                mask = self.custom_model(input_tensor)
                # Postprocesamiento de la máscara...
                mask = np.argmax(mask, axis=-1)
        
        return landmarks, mask
    
    def visualize_landmarks(self, image: np.ndarray, landmarks: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Visualiza los puntos de referencia en la imagen.

        Args:
            image: Imagen original
            landmarks: Diccionario con puntos de referencia

        Returns:
            Imagen con puntos de referencia visualizados
        """
        vis_img = image.copy()
        
        # Dibujar puntos de referencia
        for name, data in landmarks.items():
            x, y = int(data['x']), int(data['y'])
            visibility = data.get('visibility', 1.0)
            
            if visibility > 0.5:  # Solo dibujar puntos visibles
                cv2.circle(vis_img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(vis_img, name, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        # Dibujar conexiones entre puntos relevantes
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("right_shoulder", "right_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"),
            ("right_knee", "right_ankle")
        ]
        
        for start_point, end_point in connections:
            if start_point in landmarks and end_point in landmarks:
                start_x, start_y = int(landmarks[start_point]['x']), int(landmarks[start_point]['y'])
                end_x, end_y = int(landmarks[end_point]['x']), int(landmarks[end_point]['y'])
                
                cv2.line(vis_img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
                
        return vis_img
    
    def get_body_bounding_box(self, landmarks: Dict[str, Dict[str, float]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Obtiene el recuadro delimitador del cuerpo basado en los landmarks.

        Args:
            landmarks: Diccionario con puntos de referencia

        Returns:
            Tupla (x, y, width, height) o None si no hay suficientes landmarks
        """
        if not landmarks or len(landmarks) < 10:  # Necesitamos suficientes puntos
            return None
            
        x_coords = [data['x'] for data in landmarks.values() if data.get('visibility', 0) > 0.5]
        y_coords = [data['y'] for data in landmarks.values() if data.get('visibility', 0) > 0.5]
        
        if not x_coords or not y_coords:
            return None
            
        min_x, max_x = int(min(x_coords)), int(max(x_coords))
        min_y, max_y = int(min(y_coords)), int(max(y_coords))
        
        # Añadir margen
        margin = int((max_x - min_x) * 0.1)  # 10% de margen
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = max_x + margin
        max_y = max_y + margin
        
        width = max_x - min_x
        height = max_y - min_y
        
        return (min_x, min_y, width, height)
    
    def __del__(self):
        """Método para liberar recursos al destruir la instancia"""
        if hasattr(self, 'pose'):
            self.pose.close()