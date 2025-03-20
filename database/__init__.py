"""
Módulo de base de datos para el sistema de prueba virtual de ropa.
Contiene funcionalidades para conexión a la base de datos, definición
de modelos y operaciones CRUD.
"""

# Importar los módulos principales para exponerlos en la interfaz del paquete
from .db import Database, User, Measurement, ClothingItem, VirtualFitting
from .repositories import (
    UserRepository,
    MeasurementRepository,
    ClothingRepository,
    FittingRepository
)

# Definir qué clases o funciones estarán disponibles cuando se importe el paquete
__all__ = [
    'Database',
    'User',
    'Measurement',
    'ClothingItem',
    'VirtualFitting',
    'UserRepository',
    'MeasurementRepository',
    'ClothingRepository',
    'FittingRepository'
]

# Opcionalmente, se podría definir la versión del módulo
__version__ = '0.1.0'