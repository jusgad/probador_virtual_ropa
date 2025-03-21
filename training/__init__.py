"""
Módulo de entrenamiento para el probador virtual de ropa.

Este paquete contiene los componentes necesarios para entrenar
los modelos de IA utilizados en el sistema:
- Detección de poses corporales
- Segmentación de prendas de ropa
- Recomendación de tallas
"""

from training.data_preparation import DataPreparation
from training.train_pose_model import PoseModelTrainer
from training.train_segmentation_model import SegmentationModelTrainer
from training.train_size_recommender import SizeRecommenderTrainer

__all__ = [
    'DataPreparation',
    'PoseModelTrainer',
    'SegmentationModelTrainer',
    'SizeRecommenderTrainer'
]

# Información de versión del módulo de entrenamiento
__version__ = '0.1.0'