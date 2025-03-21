"""
Módulo para la evaluación de los modelos entrenados del probador virtual.

Este módulo proporciona funciones para evaluar el rendimiento de los
modelos de detección de poses, segmentación de ropa y recomendación
de tallas de manera individual o conjunta.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import joblib
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import cv2
from tqdm import tqdm
import argparse

# Importar clases de entrenamiento para reutilizar funcionalidad
from train_pose_model import PoseModelTrainer
from train_segmentation_model import SegmentationModelTrainer
from train_size_recommender import SizeRecommenderTrainer

class ModelEvaluator:
    """
    Clase para evaluar los modelos del probador virtual de ropa.
    """
    
    def __init__(self, config_path='config.json'):
        """
        Inicializa el evaluador de modelos.
        
        Args:
            config_path (str): Ruta al archivo de configuración JSON.
        """
        # Configurar logging
        self.logger = logging.getLogger(__name__)
        
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Rutas principales
        self.processed_data_path = Path(self.config['training']['processed_data_path'])
        self.model_save_path = Path(self.config['training']['model_save_path'])
        self.evaluation_path = Path(self.config['training']['model_save_path']) / 'evaluation'
        
        # Crear directorio para evaluaciones
        self.evaluation_path.mkdir(parents=True, exist_ok=True)
        
        # Timestamp para identificar esta sesión de evaluación
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Crear directorio específico para esta evaluación
        self.current_eval_path = self.evaluation_path / self.timestamp
        self.current_eval_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Inicializado evaluador de modelos. Resultados en: {self.current_eval_path}")
    
    def evaluate_pose_model(self, model_path=None, version='latest'):
        """
        Evalúa el modelo de detección de poses.
        
        Args:
            model_path (str, optional): Ruta específica al modelo a evaluar.
            version (str): Versión del modelo a evaluar ('latest', 'best', o timestamp).
            
        Returns:
            dict: Resultados de la evaluación.
        """
        self.logger.info("Evaluando modelo de detección de poses...")
        
        # Instanciar el entrenador de poses para aprovechar su funcionalidad
        pose_trainer = PoseModelTrainer()
        
        # Determinar ruta del modelo si no se especifica
        if model_path is None:
            if version == 'latest':
                # Buscar el modelo más reciente
                model_files = list((self.model_save_path / 'pose').glob('pose_model_*_final.h5'))
                if not model_files:
                    model_path = self.model_save_path / 'pose' / 'pose_model_final.h5'
                else:
                    model_path = max(model_files, key=lambda x: x.stat().st_mtime)
            elif version == 'best':
                # Buscar el mejor modelo según los resultados de entrenamiento
                results_files = list((self.model_save_path / 'pose').glob('training_results_*.json'))
                if not results_files:
                    model_path = self.model_save_path / 'pose' / 'pose_model_final.h5'
                else:
                    # Leer resultados y encontrar el mejor según val_loss
                    best_val_loss = float('inf')
                    best_timestamp = None
                    
                    for result_file in results_files:
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                            val_losses = results['training_history']['val_loss']
                            min_val_loss = min(val_losses)
                            
                            if min_val_loss < best_val_loss:
                                best_val_loss = min_val_loss
                                best_timestamp = results['timestamp']
                    
                    if best_timestamp:
                        model_path = self.model_save_path / 'pose' / f'pose_model_{best_timestamp}_final.h5'
                    else:
                        model_path = self.model_save_path / 'pose' / 'pose_model_final.h5'
            else:
                # Usar versión específica por timestamp
                model_path = self.model_save_path / 'pose' / f'pose_model_{version}_final.h5'
        
        self.logger.info(f"Evaluando modelo: {model_path}")
        
        try:
            # Cargar el modelo
            model = tf.keras.models.load_model(model_path)
            
            # Cargar dataset de test
            _, _, test_ds, _, _, test_steps = pose_trainer.load_dataset()
            
            # Evaluar en conjunto de test
            test_results = model.evaluate(test_ds, steps=test_steps)
            test_metrics = dict(zip(model.metrics_names, test_results))
            
            self.logger.info(f"Métricas en test: {test_metrics}")
            
            # Visualizar predicciones
            fig = pose_trainer.visualize_predictions(model, num_samples=5)
            
            if fig:
                fig_path = self.current_eval_path / 'pose_predictions.png'
                fig.savefig(fig_path)
                plt.close(fig)
            
            # Calcular métricas adicionales en muestras individuales
            keypoint_errors = []
            
            # Procesar algunas muestras para análisis detallado
            num_samples = min(100, test_steps * pose_trainer.batch_size)
            sample_count = 0
            
            for images, true_keypoints in test_ds:
                # Hacer predicciones batch por batch
                pred_keypoints = model.predict(images)
                
                # Calcular error por keypoint
                for i in range(len(images)):
                    if sample_count >= num_samples:
                        break
                        
                    # Error euclidiano por keypoint
                    true_kp = true_keypoints[i].numpy().reshape(-1, 2)
                    pred_kp = pred_keypoints[i].reshape(-1, 2)
                    
                    # Calcular distancia euclidiana para cada keypoint
                    distances = np.sqrt(np.sum((true_kp - pred_kp)**2, axis=1))
                    
                    # Guardar distancias
                    keypoint_errors.append(distances)
                    
                    sample_count += 1
                    
                if sample_count >= num_samples:
                    break
            
            # Calcular estadísticas por keypoint
            keypoint_errors = np.array(keypoint_errors)
            mean_errors = np.mean(keypoint_errors, axis=0)
            std_errors = np.std(keypoint_errors, axis=0)
            
            # Visualizar error por keypoint
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(mean_errors)), mean_errors, yerr=std_errors, capsize=5)
            plt.xlabel('Keypoint ID')
            plt.ylabel('Error Euclidiano Promedio')
            plt.title('Error por Keypoint')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            keypoint_error_path = self.current_eval_path / 'pose_keypoint_errors.png'
            plt.savefig(keypoint_error_path)
            plt.close()
            
            # Resultados de la evaluación
            evaluation_results = {
                'model_path': str(model_path),
                'timestamp': self.timestamp,
                'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                'keypoint_metrics': {
                    'mean_errors': mean_errors.tolist(),
                    'std_errors': std_errors.tolist(),
                    'overall_mean_error': float(np.mean(mean_errors)),
                    'max_keypoint_error': float(np.max(mean_errors)),
                    'min_keypoint_error': float(np.min(mean_errors))
                }
            }
            
            # Guardar resultados
            results_path = self.current_eval_path / 'pose_evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
                
            self.logger.info(f"Resultados de evaluación guardados en {results_path}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error al evaluar modelo de poses: {e}")
            return {'error': str(e)}
    
    def evaluate_segmentation_model(self, model_path=None, version='latest'):
        """
        Evalúa el modelo de segmentación de ropa.
        
        Args:
            model_path (str, optional): Ruta específica al modelo a evaluar.
            version (str): Versión del modelo a evaluar ('latest', 'best', o timestamp).
            
        Returns:
            dict: Resultados de la evaluación.
        """
        self.logger.info("Evaluando modelo de segmentación de ropa...")
        
        # Instanciar el entrenador de segmentación para aprovechar su funcionalidad
        seg_trainer = SegmentationModelTrainer()
        
        # Determinar ruta del modelo si no se especifica
        if model_path is None:
            if version == 'latest':
                # Buscar el modelo más reciente
                model_files = list((self.model_save_path / 'segmentation').glob('segmentation_model_*_final.h5'))
                if not model_files:
                    model_path = self.model_save_path / 'segmentation' / 'segmentation_model_final.h5'
                else:
                    model_path = max(model_files, key=lambda x: x.stat().st_mtime)
            elif version == 'best':
                # Buscar el mejor modelo según los resultados de entrenamiento
                results_files = list((self.model_save_path / 'segmentation').glob('training_results_*.json'))
                if not results_files:
                    model_path = self.model_save_path / 'segmentation' / 'segmentation_model_final.h5'
                else:
                    # Leer resultados y encontrar el mejor según val_mean_io_u
                    best_val_iou = -float('inf')
                    best_timestamp = None
                    
                    for result_file in results_files:
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                            val_ious = results['training_history']['val_mean_io_u']
                            max_val_iou = max(val_ious)
                            
                            if max_val_iou > best_val_iou:
                                best_val_iou = max_val_iou
                                best_timestamp = results['timestamp']
                    
                    if best_timestamp:
                        model_path = self.model_save_path / 'segmentation' / f'segmentation_model_{best_timestamp}_final.h5'
                    else:
                        model_path = self.model_save_path / 'segmentation' / 'segmentation_model_final.h5'
            else:
                # Usar versión específica por timestamp
                model_path = self.model_save_path / 'segmentation' / f'segmentation_model_{version}_final.h5'
        
        self.logger.info(f"Evaluando modelo: {model_path}")
        
        try:
            # Cargar el modelo
            model = tf.keras.models.load_model(model_path)
            
            # Cargar dataset de test
            _, _, test_ds, _, _, test_steps = seg_trainer.load_dataset()
            
            # Evaluar en conjunto de test
            test_results = model.evaluate(test_ds, steps=test_steps)
            test_metrics = dict(zip(model.metrics_names, test_results))
            
            self.logger.info(f"Métricas en test: {test_metrics}")
            
            # Visualizar predicciones
            fig = seg_trainer.visualize_predictions(model, num_samples=5)
            
            if fig:
                fig_path = self.current_eval_path / 'segmentation_predictions.png'
                fig.savefig(fig_path)
                plt.close(fig)
            
            # Calcular IoU por clase
            num_classes = seg_trainer.num_classes
            conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
            
            # Procesar algunas muestras para análisis detallado
            num_samples = min(100, test_steps * seg_trainer.batch_size)
            sample_count = 0
            
            for images, true_masks in test_ds:
                # Hacer predicciones batch por batch
                pred_masks = model.predict(images)
                
                # Convertir a índices de clase
                true_class = tf.argmax(true_masks, axis=-1).numpy()
                pred_class = tf.argmax(pred_masks, axis=-1).numpy()
                
                # Actualizar matriz de confusión
                for i in range(len(images)):
                    if sample_count >= num_samples:
                        break
                        
                    # Aplanar para matriz de confusión
                    true_flat = true_class[i].flatten()
                    pred_flat = pred_class[i].flatten()
                    
                    # Calcular matriz de confusión para esta imagen
                    for t, p in zip(true_flat, pred_flat):
                        conf_matrix[t, p] += 1
                    
                    sample_count += 1
                    
                if sample_count >= num_samples:
                    break
            
            # Calcular IoU por clase
            class_iou = np.zeros(num_classes)
            for i in range(num_classes):
                # Verdaderos positivos
                tp = conf_matrix[i, i]
                # Falsos negativos y falsos positivos
                fn = np.sum(conf_matrix[i, :]) - tp
                fp = np.sum(conf_matrix[:, i]) - tp
                
                # IoU = TP / (TP + FN + FP)
                if tp + fn + fp > 0:
                    class_iou[i] = tp / (tp + fn + fp)
                else:
                    class_iou[i] = 0
            
            # Visualizar IoU por clase
            class_names = ['Fondo', 'Persona', 'Ropa']  # Ajustar según el modelo
            plt.figure(figsize=(10, 6))
            bars = plt.bar(class_names, class_iou)
            
            # Añadir valores sobre las barras
            for bar, iou in zip(bars, class_iou):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{iou:.3f}',
                    ha='center',
                    va='bottom'
                )
                
            plt.xlabel('Clase')
            plt.ylabel('IoU')
            plt.title('IoU por Clase')
            plt.ylim(0, 1.0)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            iou_path = self.current_eval_path / 'segmentation_class_iou.png'
            plt.savefig(iou_path)
            plt.close()
            
            # Visualizar matriz de confusión normalizada
            plt.figure(figsize=(10, 8))
            conf_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
            sns.heatmap(conf_norm, annot=True, fmt='.3f', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.title('Matriz de Confusión Normalizada')
            
            conf_path = self.current_eval_path / 'segmentation_confusion_matrix.png'
            plt.savefig(conf_path)
            plt.close()
            
            # Resultados de la evaluación
            evaluation_results = {
                'model_path': str(model_path),
                'timestamp': self.timestamp,
                'test_metrics': {k: float(v) if isinstance(v, np.float32) else v for k, v in test_metrics.items()},
                'class_metrics': {
                    'class_iou': class_iou.tolist(),
                    'mean_iou': float(np.mean(class_iou)),
                    'class_names': class_names,
                    'confusion_matrix': conf_matrix.tolist()
                }
            }
            
            # Guardar resultados
            results_path = self.current_eval_path / 'segmentation_evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
                
            self.logger.info(f"Resultados de evaluación guardados en {results_path}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error al evaluar modelo de segmentación: {e}")
            return {'error': str(e)}
    
    def evaluate_size_recommender(self, model_path=None, version='latest'):
        """
        Evalúa el modelo de recomendación de tallas.
        
        Args:
            model_path (str, optional): Ruta específica al modelo a evaluar.
            version (str): Versión del modelo a evaluar ('latest', 'best', o timestamp).
            
        Returns:
            dict: Resultados de la evaluación.
        """
        self.logger.info("Evaluando modelo de recomendación de tallas...")
        
        # Instanciar el entrenador de recomendación para aprovechar su funcionalidad
        size_trainer = SizeRecommenderTrainer()
        
        # Determinar ruta del modelo si no se especifica
        if model_path is None:
            if version == 'latest':
                # Buscar el modelo más reciente
                model_files = list((self.model_save_path / 'size_recommender').glob('size_model_*_final.h5'))
                if not model_files:
                    model_path = self.model_save_path / 'size_recommender' / 'size_model_final.h5'
                else:
                    model_path = max(model_files, key=lambda x: x.stat().st_mtime)
            elif version == 'best':
                # Buscar el mejor modelo según los resultados de entrenamiento
                results_files = list((self.model_save_path / 'size_recommender').glob('training_results_*.json'))
                if not results_files:
                    model_path = self.model_save_path / 'size_recommender' / 'size_model_final.h5'
                else:
                    # Leer resultados y encontrar el mejor según val_loss
                    best_val_loss = float('inf')
                    best_timestamp = None
                    
                    for result_file in results_files:
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                            val_losses = results['training_history']['val_loss']
                            min_val_loss = min(val_losses)
                            
                            if min_val_loss < best_val_loss:
                                best_val_loss = min_val_loss
                                best_timestamp = results['timestamp']
                    
                    if best_timestamp:
                        model_path = self.model_save_path / 'size_recommender' / f'size_model_{best_timestamp}_final.h5'
                    else:
                        model_path = self.model_save_path / 'size_recommender' / 'size_model_final.h5'
            else:
                # Usar versión específica por timestamp
                model_path = self.model_save_path / 'size_recommender' / f'size_model_{version}_final.h5'
        
        self.logger.info(f"Evaluando modelo: {model_path}")
        
        try:
            # Usar la funcionalidad de evaluación del entrenador
            evaluation_results = size_trainer.evaluate_model(model_path=model_path)
            
            # Copiar visualizaciones al directorio de evaluación actual
            source_dir = Path(evaluation_results['visualization_dir'])
            for file in source_dir.glob('*.png'):
                target_path = self.current_eval_path / file.name
                if not target_path.exists():
                    target_path.write_bytes(file.read_bytes())
            
            # Actualizar ruta de resultados
            evaluation_results['timestamp'] = self.timestamp
            evaluation_results['model_path'] = str(model_path)
            
            # Guardar resultados
            results_path = self.current_eval_path / 'size_recommender_evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
                
            self.logger.info(f"Resultados de evaluación guardados en {results_path}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error al evaluar modelo de recomendación de tallas: {e}")
            return {'error': str(e)}
    
    def evaluate_all_models(self, versions=None):
        """
        Evalúa todos los modelos del sistema.
        
        Args:
            versions (dict, optional): Diccionario con versiones específicas para cada modelo.
                Ejemplo: {'pose': 'latest', 'segmentation': '20230420-123456', 'size': 'best'}
                
        Returns:
            dict: Resultados de la evaluación para todos los modelos.
        """
        self.logger.info("Evaluando todos los modelos del sistema...")
        
        # Configurar versiones por defecto
        if versions is None:
            versions = {
                'pose': 'latest',
                'segmentation': 'latest',
                'size': 'latest'
            }
        
        # Evaluar cada modelo
        results = {}
        
        # Modelo de poses
        pose_version = versions.get('pose', 'latest')
        self.logger.info(f"Evaluando modelo de poses versión: {pose_version}")
        results['pose'] = self.evaluate_pose_model(version=pose_version)
        
        # Modelo de segmentación
        seg_version = versions.get('segmentation', 'latest')
        self.logger.info(f"Evaluando modelo de segmentación versión: {seg_version}")
        results['segmentation'] = self.evaluate_segmentation_model(version=seg_version)
        
        # Modelo de recomendación de tallas
        size_version = versions.get('size', 'latest')
        self.logger.info(f"Evaluando modelo de recomendación de tallas versión: {size_version}")
        results['size'] = self.evaluate_size_recommender(version=size_version)
        
        # Generar informe resumen
        summary = {
            'timestamp': self.timestamp,
            'versions_evaluated': versions,
            'summary_metrics': {
                'pose': results['pose'].get('test_metrics', {}),
                'segmentation': results['segmentation'].get('test_metrics', {}),
                'size': results['size'].get('overall_metrics', {})
            }
        }
        
        # Guardar resumen
        summary_path = self.current_eval_path / 'evaluation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Crear visualización resumen
        self.create_summary_visualization(summary)
        
        self.logger.info(f"Evaluación de todos los modelos completada. Resumen en {summary_path}")
        
        return results
    
    def create_summary_visualization(self, summary):
        """
        Crea una visualización resumen de los resultados de evaluación.
        
        Args:
            summary (dict): Resumen de la evaluación.
        """
        # Crear figura con múltiples subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Métricas del modelo de pose
        pose_metrics = summary['summary_metrics']['pose']
        if pose_metrics:
            # Extraer métricas relevantes
            metrics = list(pose_metrics.keys())
            values = [pose_metrics[m] for m in metrics]
            
            # Graficar
            axs[0].bar(metrics, values)
            axs[0].set_title('Métricas del Modelo de Poses')
            axs[0].set_ylim(0, max(values) * 1.2)
            axs[0].set_xticklabels(metrics, rotation=45, ha='right')
            axs[0].grid(True, linestyle='--', alpha=0.7)
            
            # Añadir valores sobre las barras
            for i, v in enumerate(values):
                axs[0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. Métricas del modelo de segmentación
        seg_metrics = summary['summary_metrics']['segmentation']
        if seg_metrics:
            # Extraer métricas relevantes (enfocarse en IoU)
            if 'mean_io_u' in seg_metrics:
                metrics = ['mean_io_u', 'accuracy']
                values = [seg_metrics.get(m, 0) for m in metrics]
                
                # Graficar
                axs[1].bar(metrics, values)
                axs[1].set_title('Métricas del Modelo de Segmentación')
                axs[1].set_ylim(0, 1.0)
                axs[1].grid(True, linestyle='--', alpha=0.7)
                
                # Añadir valores sobre las barras
                for i, v in enumerate(values):
                    axs[1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 3. Métricas del modelo de recomendación de tallas
        size_metrics = summary['summary_metrics']['size']
        if size_metrics:
            # Extraer métricas relevantes
            if 'accuracy_by_category' in size_metrics:
                categories = list(size_metrics['accuracy_by_category'].keys())
                accuracies = [size_metrics['accuracy_by_category'][c] for c in categories]
                
                # Graficar
                axs[2].bar(categories, accuracies)
                axs[2].set_title('Precisión por Categoría (Recomendador de Tallas)')
                axs[2].set_ylim(0, 1.0)
                axs[2].set_xticklabels(categories, rotation=45, ha='right')
                axs[2].grid(True, linestyle='--', alpha=0.7)
                
                # Añadir valores sobre las barras
                for i, v in enumerate(accuracies):
                    axs[2].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        # Guardar figura
        summary_path = self.current_eval_path / 'evaluation_summary_visualization.png'
        plt.savefig(summary_path)
        plt.close()

def main():
    """Función principal para ejecutar la evaluación desde línea de comandos."""
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Evaluar modelos del probador virtual de ropa')
    
    parser.add_argument('--config', type=str, default='config.json',
                        help='Ruta al archivo de configuración')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        choices=['all', 'pose', 'segmentation', 'size'],
                        default=['all'],
                        help='Modelos a evaluar (all, pose, segmentation, size)')
    
    parser.add_argument('--pose-version', type=str, default='latest',
                        help='Versión del modelo de poses a evaluar')
    
    parser.add_argument('--segmentation-version', type=str, default='latest',
                        help='Versión del modelo de segmentación a evaluar')
    
    parser.add_argument('--size-version', type=str, default='latest',
                        help='Versión del modelo de recomendación de tallas a evaluar')
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear evaluador
    evaluator = ModelEvaluator(config_path=args.config)
    
    # Determinar modelos a evaluar
    models_to_eval = args.models
    if 'all' in models_to_eval:
        models_to_eval = ['pose', 'segmentation', 'size']
    
    # Configurar versiones
    versions = {
        'pose': args.pose_version,
        'segmentation': args.segmentation_version,
        'size': args.size_version
    }
    
    # Evaluar modelos individualmente o todos juntos
    if len(models_to_eval) == 3:
        # Evaluar todos
        evaluator.evaluate_all_models(versions=versions)
    else:
        # Evaluar modelos individualmente
        for model in models_to_eval:
            if model == 'pose':
                evaluator.evaluate_pose_model(version=versions['pose'])
            elif model == 'segmentation':
                evaluator.evaluate_segmentation_model(version=versions['segmentation'])
            elif model == 'size':
                evaluator.evaluate_size_recommender(version=versions['size'])

if __name__ == "__main__":
    main()