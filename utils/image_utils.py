"""
Utilidades para procesamiento de imágenes en el sistema de prueba virtual.
Proporciona funciones para manipular, transformar y combinar imágenes.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union, List
import math
import os
import logging

# Configurar logging
logger = logging.getLogger(__name__)


def resize_image(image: np.ndarray, width: Optional[int] = None, 
                height: Optional[int] = None, keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Redimensiona una imagen a las dimensiones especificadas.
    
    Args:
        image: Imagen como array numpy
        width: Ancho deseado (opcional)
        height: Alto deseado (opcional)
        keep_aspect_ratio: Si se debe mantener la relación de aspecto
        
    Returns:
        Imagen redimensionada
    """
    if image is None:
        logger.error("Image is None in resize_image")
        return np.zeros((10, 10, 3), dtype=np.uint8)  # Imagen vacía como fallback
        
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
        
    if keep_aspect_ratio:
        if width is None:
            aspect = w / h
            width = int(height * aspect)
        elif height is None:
            aspect = h / w
            height = int(width * aspect)
        else:
            # Si ambos son proporcionados, se ajusta para no distorsionar
            h_ratio = height / h
            w_ratio = width / w
            ratio = min(h_ratio, w_ratio)
            width = int(w * ratio)
            height = int(h * ratio)
    
    # Asegurar valores mínimos para evitar errores
    width = max(width, 1)
    height = max(height, 1)
    
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def resize_image_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """
    Redimensiona una imagen a una altura específica manteniendo la relación de aspecto.
    
    Args:
        image: Imagen como array numpy
        target_height: Altura deseada en píxeles
        
    Returns:
        Imagen redimensionada
    """
    h, w = image.shape[:2]
    ratio = target_height / h
    target_width = int(w * ratio)
    
    return resize_image(image, target_width, target_height)


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Recorta una porción de la imagen.
    
    Args:
        image: Imagen como array numpy
        x: Coordenada X desde donde iniciar el recorte
        y: Coordenada Y desde donde iniciar el recorte
        width: Ancho del recorte
        height: Alto del recorte
        
    Returns:
        Imagen recortada
    """
    h, w = image.shape[:2]
    
    # Asegurar que las coordenadas estén dentro de los límites
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    width = min(width, w - x)
    height = min(height, h - y)
    
    return image[y:y+height, x:x+width].copy()


def overlay_images(background: np.ndarray, foreground: np.ndarray, 
                  position: Tuple[int, int] = None) -> np.ndarray:
    """
    Superpone una imagen sobre otra respetando la transparencia.
    
    Args:
        background: Imagen de fondo
        foreground: Imagen a superponer (con canal alfa)
        position: Posición (x, y) donde colocar la imagen superpuesta
                 Si es None, se centra
        
    Returns:
        Imagen combinada
    """
    if background is None or foreground is None:
        logger.error("One of the images is None in overlay_images")
        return background if background is not None else foreground
    
    # Asegurar que el fondo tenga canal alfa
    if background.shape[2] == 3:
        bg_h, bg_w = background.shape[:2]
        background_rgba = np.zeros((bg_h, bg_w, 4), dtype=np.uint8)
        background_rgba[:, :, :3] = background
        background_rgba[:, :, 3] = 255
        background = background_rgba
    
    # Asegurar que el primer plano tenga canal alfa
    if foreground.shape[2] == 3:
        fg_h, fg_w = foreground.shape[:2]
        foreground_rgba = np.zeros((fg_h, fg_w, 4), dtype=np.uint8)
        foreground_rgba[:, :, :3] = foreground
        foreground_rgba[:, :, 3] = 255
        foreground = foreground_rgba
    
    # Obtener dimensiones
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]
    
    # Determinar posición si no se proporciona
    if position is None:
        x = (bg_w - fg_w) // 2
        y = (bg_h - fg_h) // 2
    else:
        x, y = position
    
    # Verificar límites
    if x >= bg_w or y >= bg_h or x + fg_w <= 0 or y + fg_h <= 0:
        return background  # Fuera de límites
    
    # Ajustar coordenadas si parte de la imagen queda fuera
    x_offset = max(0, -x)
    y_offset = max(0, -y)
    x = max(0, x)
    y = max(0, y)
    
    # Determinar el tamaño de la región de superposición
    w = min(fg_w - x_offset, bg_w - x)
    h = min(fg_h - y_offset, bg_h - y)
    
    if w <= 0 or h <= 0:
        return background  # No hay región de superposición
    
    # Extraer regiones de superposición
    alpha = foreground[y_offset:y_offset+h, x_offset:x_offset+w, 3] / 255.0
    alpha = alpha.reshape(h, w, 1)
    
    # Aplicar mezcla de canal alfa
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            background[y:y+h, x:x+w, c] * (1 - alpha[:, :, 0]) + 
            foreground[y_offset:y_offset+h, x_offset:x_offset+w, c] * alpha[:, :, 0]
        ).astype(np.uint8)
    
    # Actualizar también el canal alfa del fondo si es necesario
    if background.shape[2] == 4:
        background[y:y+h, x:x+w, 3] = np.maximum(
            background[y:y+h, x:x+w, 3],
            foreground[y_offset:y_offset+h, x_offset:x_offset+w, 3]
        )
    
    return background


def warp_image(image: np.ndarray, source_points: np.ndarray, 
              target_points: np.ndarray, output_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Deforma una imagen usando una transformación de perspectiva.
    
    Args:
        image: Imagen a deformar
        source_points: Puntos de origen en la imagen (4 puntos como mínimo)
        target_points: Puntos de destino
        output_size: Tamaño de la imagen resultante (ancho, alto)
        
    Returns:
        Imagen deformada
    """
    if len(source_points) < 4 or len(target_points) < 4:
        logger.error("Need at least 4 points for perspective transform")
        return image
    
    if output_size is None:
        output_size = (image.shape[1], image.shape[0])
        
    # Convertir puntos a formato numpy si no lo están ya
    if not isinstance(source_points, np.ndarray):
        source_points = np.array(source_points, dtype=np.float32)
    if not isinstance(target_points, np.ndarray):
        target_points = np.array(target_points, dtype=np.float32)
        
    # Calcular matriz de transformación
    transform_matrix = cv2.getPerspectiveTransform(source_points, target_points)
    
    # Aplicar transformación
    warped = cv2.warpPerspective(
        image, 
        transform_matrix, 
        output_size, 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0, 0) if image.shape[2] == 4 else (0, 0, 0)
    )
    
    return warped


def apply_transparency_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Aplica una máscara de transparencia a una imagen.
    
    Args:
        image: Imagen a procesar
        mask: Máscara (0-255 donde 0 es completamente transparente)
        
    Returns:
        Imagen con transparencia aplicada
    """
    # Asegurar que la imagen tenga canal alfa
    if image.shape[2] == 3:
        h, w = image.shape[:2]
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[:, :, :3] = image
        result[:, :, 3] = 255
    else:
        result = image.copy()
    
    # Asegurar que la máscara tenga las dimensiones correctas
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Si la máscara es de 3 canales, convertir a 1 canal
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Aplicar la máscara al canal alfa
    result[:, :, 3] = mask
    
    return result


def blend_images(image1: np.ndarray, image2: np.ndarray, 
                alpha: float = 0.5) -> np.ndarray:
    """
    Mezcla dos imágenes con el factor alpha especificado.
    
    Args:
        image1: Primera imagen
        image2: Segunda imagen
        alpha: Factor de mezcla (0.0 - 1.0 donde 1.0 muestra solo image1)
        
    Returns:
        Imagen mezclada
    """
    # Verificar que las imágenes tengan el mismo tamaño
    if image1.shape[:2] != image2.shape[:2]:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    # Asegurar que ambas imágenes tengan el mismo número de canales
    if image1.shape[2] != image2.shape[2]:
        if image1.shape[2] == 3 and image2.shape[2] == 4:
            # Convertir image1 a RGBA
            temp = np.zeros((image1.shape[0], image1.shape[1], 4), dtype=np.uint8)
            temp[:, :, :3] = image1
            temp[:, :, 3] = 255
            image1 = temp
        elif image1.shape[2] == 4 and image2.shape[2] == 3:
            # Convertir image2 a RGBA
            temp = np.zeros((image2.shape[0], image2.shape[1], 4), dtype=np.uint8)
            temp[:, :, :3] = image2
            temp[:, :, 3] = 255
            image2 = temp
    
    # Realizar la mezcla
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = None, 
                    normalize: bool = True) -> np.ndarray:
    """
    Preprocesa una imagen para utilizarla en modelos de ML/DL.
    
    Args:
        image: Imagen a preprocesar
        target_size: Tamaño objetivo (ancho, alto)
        normalize: Si se debe normalizar los valores de píxeles (0-1)
        
    Returns:
        Imagen preprocesada
    """
    # Redimensionar si se especifica un tamaño
    if target_size:
        image = resize_image(image, target_size[0], target_size[1])
    
    # Convertir a RGB si está en BGR (OpenCV)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalizar si se solicita
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    return image


def create_white_background(width: int, height: int) -> np.ndarray:
    """
    Crea una imagen con fondo blanco.
    
    Args:
        width: Ancho deseado
        height: Alto deseado
        
    Returns:
        Imagen con fondo blanco
    """
    return np.ones((height, width, 3), dtype=np.uint8) * 255


def extract_contours(image: np.ndarray, threshold_value: int = 127, 
                     min_area: int = 100) -> List[np.ndarray]:
    """
    Extrae los contornos de una imagen.
    
    Args:
        image: Imagen de entrada
        threshold_value: Valor para umbralización
        min_area: Área mínima para considerar un contorno válido
        
    Returns:
        Lista de contornos encontrados
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Umbralizar
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar por área mínima
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]


def draw_landmarks(image: np.ndarray, landmarks: dict, 
                  radius: int = 5, color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = -1, text: bool = True) -> np.ndarray:
    """
    Dibuja landmarks en una imagen.
    
    Args:
        image: Imagen donde dibujar
        landmarks: Diccionario con landmarks {nombre: {x, y, ...}}
        radius: Radio de los círculos
        color: Color en formato BGR
        thickness: Grosor de línea (-1 para rellenar)
        text: Si se debe mostrar el nombre del landmark
        
    Returns:
        Imagen con landmarks dibujados
    """
    result = image.copy()
    
    for name, data in landmarks.items():
        x, y = int(data['x']), int(data['y'])
        visibility = data.get('visibility', 1.0)
        
        if visibility > 0.3:  # Solo dibujar puntos visibles
            cv2.circle(result, (x, y), radius, color, thickness)
            
            if text:
                cv2.putText(
                    result, name, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
    
    return result


def rotate_image(image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Rota una imagen alrededor de un centro especificado.
    
    Args:
        image: Imagen a rotar
        angle: Ángulo de rotación en grados
        center: Centro de rotación (si es None, se usa el centro de la imagen)
        
    Returns:
        Imagen rotada
    """
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # Calcular matriz de rotación
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Aplicar transformación
    rotated = cv2.warpAffine(
        image, M, (w, h), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0, 0) if image.shape[2] == 4 else (0, 0, 0)
    )
    
    return rotated


def adjust_image_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Ajusta el brillo de una imagen.
    
    Args:
        image: Imagen a ajustar
        factor: Factor de brillo (>1 aumenta, <1 disminuye)
        
    Returns:
        Imagen con brillo ajustado
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    
    # Escalar el canal V
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    hsv = np.array(hsv, dtype=np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return result


def save_image(image: np.ndarray, filepath: str, create_dirs: bool = True) -> bool:
    """
    Guarda una imagen en disco.
    
    Args:
        image: Imagen a guardar
        filepath: Ruta donde guardar
        create_dirs: Si se deben crear directorios que no existan
        
    Returns:
        True si la operación fue exitosa
    """
    try:
        if create_dirs:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Determinar si guardar con transparencia
        if image.shape[2] == 4:
            # Imagen con canal alfa (PNG)
            return cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            # Imagen normal
            return cv2.imwrite(filepath, image)
    except Exception as e:
        logger.error(f"Error saving image to {filepath}: {str(e)}")
        return False


def load_image(filepath: str, with_alpha: bool = True) -> Optional[np.ndarray]:
    """
    Carga una imagen desde disco.
    
    Args:
        filepath: Ruta de la imagen
        with_alpha: Si se debe cargar con canal alfa
        
    Returns:
        Imagen cargada o None si hay error
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Image file not found: {filepath}")
            return None
        
        if with_alpha:
            return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        else:
            return cv2.imread(filepath, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error loading image from {filepath}: {str(e)}")
        return None