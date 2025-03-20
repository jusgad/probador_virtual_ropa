"""
Utilidades para el sistema de prueba virtual de ropa.
Contiene funciones y clases auxiliares para procesamiento de imágenes
y manipulación de datos.
"""

# Importar las funciones y clases principales para exponerlas en la interfaz del paquete
from .image_utils import (
    resize_image, 
    overlay_images, 
    warp_image, 
    preprocess_image,
    apply_transparency_mask,
    blend_images
)

from .data_utils import (
    parse_size_chart,
    validate_measurements,
    format_measurements_output,
    calculate_bmi,
    generate_filename
)

# Definir qué funciones o clases estarán disponibles cuando se importe el paquete
__all__ = [
    # Utilidades de imagen
    'resize_image',
    'overlay_images',
    'warp_image',
    'preprocess_image',
    'apply_transparency_mask',
    'blend_images',
    
    # Utilidades de datos
    'parse_size_chart',
    'validate_measurements',
    'format_measurements_output',
    'calculate_bmi',
    'generate_filename'
]

# Opcionalmente, se podría definir la versión del módulo
__version__ = '0.1.0'