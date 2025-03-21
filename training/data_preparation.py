"""
Módulo para la preparación de datos de entrenamiento.

Este módulo se encarga de procesar los datos crudos y convertirlos
en formatos adecuados para el entrenamiento de los modelos de
detección de poses, segmentación de ropa y recomendación de tallas.
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
import shutil
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataPreparation:
    """
    Clase para preparar datos de entrenamiento para todos los modelos
    del probador virtual de ropa.
    """
    
    def __init__(self, config_path='config.json'):
        """
        Inicializa el preparador de datos.
        
        Args:
            config_path (str): Ruta al archivo de configuración JSON.
        """
        # Configurar logging
        self.logger = logging.getLogger(__name__)
        
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Rutas principales
        self.raw_data_path = Path(self.config['training']['raw_data_path'])
        self.processed_data_path = Path(self.config['training']['processed_data_path'])
        
        # Crear directorios si no existen
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        (self.processed_data_path / 'pose_data').mkdir(exist_ok=True)
        (self.processed_data_path / 'segmentation_data').mkdir(exist_ok=True)
        (self.processed_data_path / 'size_data').mkdir(exist_ok=True)
        
        self.logger.info(f"Datos brutos: {self.raw_data_path}")
        self.logger.info(f"Datos procesados: {self.processed_data_path}")
    
    def prepare_pose_dataset(self):
        """
        Prepara el dataset para entrenamiento del modelo de detección de poses.
        
        Procesa las imágenes y anotaciones de landmarks corporales, y las divide
        en conjuntos de entrenamiento, validación y prueba.
        
        Returns:
            dict: Diccionario con las divisiones train/val/test.
        """
        self.logger.info("Preparando dataset para modelo de poses...")
        
        # Verificar que exista el directorio de anotaciones
        annotations_path = self.raw_data_path / 'annotations' / 'pose_annotations.json'
        if not annotations_path.exists():
            self.logger.error(f"No se encontró el archivo de anotaciones: {annotations_path}")
            raise FileNotFoundError(f"No se encontró el archivo: {annotations_path}")
        
        # Cargar anotaciones
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        self.logger.info(f"Cargadas {len(annotations)} anotaciones de poses")
        
        # Dividir en train/val/test
        image_ids = list(annotations.keys())
        train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
        train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=42)
        
        # Guardar divisiones
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        
        splits_path = self.processed_data_path / 'pose_data' / 'splits.json'
        with open(splits_path, 'w') as f:
            json.dump(splits, f)
        
        # Procesar y copiar imágenes a directorios específicos
        for split, ids in splits.items():
            split_dir = self.processed_data_path / 'pose_data' / split
            split_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Procesando {len(ids)} imágenes para división '{split}'")
            
            for img_id in tqdm(ids, desc=f"Procesando {split}"):
                # Verificar que exista la imagen
                img_path = self.raw_data_path / 'images' / f"{img_id}.jpg"
                if not img_path.exists():
                    self.logger.warning(f"No se encontró la imagen: {img_path}")
                    continue
                
                # Leer y procesar imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    self.logger.warning(f"No se pudo leer la imagen: {img_path}")
                    continue
                
                # Crear directorio para datos procesados si no existe
                img_target_path = split_dir / f"{img_id}.jpg"
                
                # Redimensionar si es necesario
                img_size = self.config['training']['pose'].get('img_size', 224)
                if img.shape[0] != img_size or img.shape[1] != img_size:
                    img = cv2.resize(img, (img_size, img_size))
                
                # Guardar imagen procesada
                cv2.imwrite(str(img_target_path), img)
                
                # Crear archivo de anotaciones específico para esta imagen
                annotation_target_path = split_dir / f"{img_id}_keypoints.json"
                with open(annotation_target_path, 'w') as f:
                    json.dump(annotations[img_id], f)
        
        # Crear archivo de metadatos
        metadata = {
            'num_train': len(train_ids),
            'num_val': len(val_ids),
            'num_test': len(test_ids),
            'img_size': img_size,
            'num_keypoints': self.config['training']['pose'].get('num_keypoints', 17)
        }
        
        metadata_path = self.processed_data_path / 'pose_data' / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        self.logger.info(f"Dataset de poses preparado: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
        return splits
    
    def prepare_segmentation_dataset(self):
        """
        Prepara dataset para entrenamiento del modelo de segmentación de ropa.
        
        Procesa imágenes y sus máscaras correspondientes, y las divide en
        conjuntos de entrenamiento, validación y prueba.
        
        Returns:
            dict: Diccionario con las divisiones train/val/test.
        """
        self.logger.info("Preparando dataset para modelo de segmentación...")
        
        # Verificar que existan los directorios necesarios
        images_path = self.raw_data_path / 'segmentation' / 'images'
        masks_path = self.raw_data_path / 'segmentation' / 'masks'
        
        if not images_path.exists() or not masks_path.exists():
            self.logger.error(f"No se encontraron los directorios: {images_path} o {masks_path}")
            raise FileNotFoundError(f"Directorios no encontrados: {images_path} o {masks_path}")
        
        # Listar archivos de imágenes
        image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.logger.info(f"Encontradas {len(image_files)} imágenes para segmentación")
        
        # Verificar que existan las máscaras correspondientes
        valid_files = []
        for img_file in image_files:
            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            if (masks_path / mask_file).exists():
                valid_files.append(img_file)
            else:
                self.logger.warning(f"No se encontró la máscara para: {img_file}")
        
        self.logger.info(f"Seleccionadas {len(valid_files)} imágenes con máscara válida")
        
        # Dividir en train/val/test
        train_files, test_files = train_test_split(valid_files, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)
        
        # Guardar divisiones
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        splits_path = self.processed_data_path / 'segmentation_data' / 'splits.json'
        with open(splits_path, 'w') as f:
            json.dump(splits, f)
        
        # Procesar y copiar imágenes y máscaras a directorios específicos
        img_size = self.config['training']['segmentation'].get('img_size', 256)
        num_classes = self.config['training']['segmentation'].get('num_classes', 3)
        
        for split, files in splits.items():
            # Crear directorios para datos procesados
            images_split_dir = self.processed_data_path / 'segmentation_data' / split / 'images'
            masks_split_dir = self.processed_data_path / 'segmentation_data' / split / 'masks'
            
            images_split_dir.mkdir(parents=True, exist_ok=True)
            masks_split_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Procesando {len(files)} imágenes para división '{split}'")
            
            for img_file in tqdm(files, desc=f"Procesando {split}"):
                # Rutas de origen
                img_path = images_path / img_file
                mask_path = masks_path / img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
                
                # Rutas de destino
                img_target_path = images_split_dir / img_file
                mask_target_path = masks_split_dir / img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
                
                # Procesar imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    self.logger.warning(f"No se pudo leer la imagen: {img_path}")
                    continue
                
                # Redimensionar imagen
                img = cv2.resize(img, (img_size, img_size))
                
                # Procesar máscara
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    self.logger.warning(f"No se pudo leer la máscara: {mask_path}")
                    continue
                
                # Redimensionar máscara usando interpolación nearest para preservar las clases
                mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                
                # Verificar valores de la máscara
                unique_values = np.unique(mask)
                if max(unique_values) >= num_classes:
                    self.logger.warning(f"Valores inesperados en la máscara {mask_path}: {unique_values}")
                
                # Guardar imagen y máscara procesadas
                cv2.imwrite(str(img_target_path), img)
                cv2.imwrite(str(mask_target_path), mask)
        
        # Crear archivo de metadatos
        metadata = {
            'num_train': len(train_files),
            'num_val': len(val_files),
            'num_test': len(test_files),
            'img_size': img_size,
            'num_classes': num_classes,
            'class_mapping': {
                '0': 'background',
                '1': 'person',
                '2': 'clothing'
            }
        }
        
        metadata_path = self.processed_data_path / 'segmentation_data' / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        self.logger.info(f"Dataset de segmentación preparado: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        return splits
    
    def prepare_size_dataset(self):
        """
        Prepara dataset para entrenamiento del recomendador de tallas.
        
        Procesa datos de medidas corporales y preferencias de tallas de usuarios,
        y los divide en conjuntos de entrenamiento, validación y prueba.
        """
        self.logger.info("Preparando dataset para recomendador de tallas...")
        
        # Verificar que existan los archivos necesarios
        measurements_file = self.raw_data_path / 'size_data' / 'user_measurements.csv'
        preferences_file = self.raw_data_path / 'size_data' / 'size_preferences.csv'
        
        if not measurements_file.exists() or not preferences_file.exists():
            self.logger.error(f"No se encontraron los archivos: {measurements_file} o {preferences_file}")
            raise FileNotFoundError(f"Archivos no encontrados: {measurements_file} o {preferences_file}")
        
        # Cargar datos
        try:
            measurements_df = pd.read_csv(measurements_file)
            preferences_df = pd.read_csv(preferences_file)
            
            self.logger.info(f"Cargadas {len(measurements_df)} medidas y {len(preferences_df)} preferencias")
        except Exception as e:
            self.logger.error(f"Error al cargar los archivos CSV: {e}")
            raise
        
        # Verificar columnas necesarias
        required_cols_measurements = ['user_id', 'height', 'weight', 'chest', 'waist', 'hips']
        required_cols_preferences = ['user_id', 'brand', 'category', 'preferred_size']
        
        for col in required_cols_measurements:
            if col not in measurements_df.columns:
                self.logger.error(f"Columna requerida '{col}' no encontrada en {measurements_file}")
                raise ValueError(f"Columna requerida '{col}' no encontrada")
        
        for col in required_cols_preferences:
            if col not in preferences_df.columns:
                self.logger.error(f"Columna requerida '{col}' no encontrada en {preferences_file}")
                raise ValueError(f"Columna requerida '{col}' no encontrada")
        
        # Unir dataframes
        df = pd.merge(measurements_df, preferences_df, on='user_id')
        self.logger.info(f"Dataset combinado contiene {len(df)} registros")
        
        # Preprocesamiento
        # Renombrar columnas para consistencia
        cols_mapping = {}
        for col in measurements_df.columns:
            if col != 'user_id':
                cols_mapping[col] = f"measurement_{col}"
        
        df = df.rename(columns=cols_mapping)
        
        # Convertir variables categóricas
        cat_cols = ['brand', 'category', 'preferred_size']
        for col in cat_cols:
            df[f"size_{col}"] = df[col]
            df = df.drop(columns=[col])
        
        # Dividir en train/val/test
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)
        
        # Guardar dataframes procesados
        train_df.to_csv(self.processed_data_path / 'size_data' / 'train.csv', index=False)
        val_df.to_csv(self.processed_data_path / 'size_data' / 'val.csv', index=False)
        test_df.to_csv(self.processed_data_path / 'size_data' / 'test.csv', index=False)
        
        # Guardar metadatos
        feature_cols = [col for col in df.columns if col.startswith('measurement_')]
        target_cols = [col for col in df.columns if col.startswith('size_')]
        
        metadata = {
            'num_train': len(train_df),
            'num_val': len(val_df),
            'num_test': len(test_df),
            'feature_columns': feature_cols,
            'target_columns': target_cols,
            'categorical_columns': [col for col in df.columns if col.startswith('size_')],
            'numerical_columns': [col for col in df.columns if col.startswith('measurement_')]
        }
        
        metadata_path = self.processed_data_path / 'size_data' / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        self.logger.info(f"Dataset de tallas preparado: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    def create_data_summary(self):
        """
        Crea un resumen de los datos de entrenamiento disponibles.
        
        Genera un archivo JSON con información sobre todos los datasets.
        """
        summary = {
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'datasets': {}
        }
        
        # Resumen de datos de poses
        pose_metadata_path = self.processed_data_path / 'pose_data' / 'metadata.json'
        if pose_metadata_path.exists():
            with open(pose_metadata_path, 'r') as f:
                summary['datasets']['pose'] = json.load(f)
        
        # Resumen de datos de segmentación
        seg_metadata_path = self.processed_data_path / 'segmentation_data' / 'metadata.json'
        if seg_metadata_path.exists():
            with open(seg_metadata_path, 'r') as f:
                summary['datasets']['segmentation'] = json.load(f)
        
        # Resumen de datos de tallas
        size_metadata_path = self.processed_data_path / 'size_data' / 'metadata.json'
        if size_metadata_path.exists():
            with open(size_metadata_path, 'r') as f:
                summary['datasets']['size_recommender'] = json.load(f)
        
        # Guardar resumen
        summary_path = self.processed_data_path / 'data_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Resumen de datos creado en {summary_path}")
        return summary
    
    def run_all(self):
        """
        Ejecuta todos los procesos de preparación de datos.
        
        Returns:
            dict: Resumen de los datasets preparados.
        """
        try:
            self.prepare_pose_dataset()
            self.prepare_segmentation_dataset()
            self.prepare_size_dataset()
            summary = self.create_data_summary()
            self.logger.info("Preparación de datos completa.")
            return summary
        except Exception as e:
            self.logger.error(f"Error durante la preparación de datos: {e}")
            raise

if __name__ == "__main__":
    # Configurar logging básico si se ejecuta como script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar preparación de datos
    data_prep = DataPreparation()
    data_prep.run_all()