"""
Módulo central para el sistema de prueba virtual de ropa.
Contiene las funcionalidades principales para detección corporal,
mediciones, ajuste de prendas y recomendación de tallas.
"""

# Importar los módulos principales para exponerlos en la interfaz del paquete
from .body_detector import BodyDetector
from .measurement import MeasurementCalculator
from .clothing_fitter import ClothingFitter
from .size_recommender import SizeRecommender

# Definir qué clases o funciones estarán disponibles cuando se importe el paquete
__all__ = [
    'BodyDetector',
    'MeasurementCalculator',
    'ClothingFitter',
    'SizeRecommender'
]

# Opcionalmente, se podría definir la versión del módulo
__version__ = '0.1.0'