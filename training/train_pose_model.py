"""
Módulo para entrenar el modelo de detección de poses corporales.

Este módulo implementa la funcionalidad para entrenar un modelo
de deep learning que detecta landmarks corporales en imágenes.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

class PoseModelTrainer:
    """
    Clase para entrenar el modelo de detección de poses corporales.
    """
    
    def __init__(self, config_path='config.json'):
        """
        Inicializa el entrenador del modelo de poses.
        
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
        self.log_path = Path(self.config['training']['log_path'])
        
        # Parámetros de entrenamiento
        self.train_config = self.config['training']['pose']
        self.batch_size = self.train_config.get('batch_size', 16)
        self.epochs = self.train_config.get('epochs', 50)
        self.img_size = self.train_config.get('img_size', 224)
        self.num_keypoints = self.train_config.get('num_keypoints', 17)
        self.learning_rate = self.train_config.get('learning_rate', 0.001)
        self.use_augmentation = self.train_config.get('augmentation', True)
        
        # Crear directorios necesarios
        self.model_save_path = self.model_save_path / 'pose'
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.log_path = self.log_path / 'pose'
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Timestamp para identificar esta sesión de entrenamiento
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.logger.info(f"Inicializado entrenador de modelo de poses con tamaño de imagen {self.img_size}x{self.img_size}")
    
    def load_metadata(self):
        """
        Carga los metadatos del dataset preparado.
        
        Returns:
            dict: Metadatos del dataset de poses.
        """
        metadata_path = self.processed_data_path / 'pose_data' / 'metadata.json'
        
        if not metadata_path.exists():
            self.logger.error(f"No se encontró el archivo de metadatos: {metadata_path}")
            raise FileNotFoundError(f"No se encontró el archivo: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        self.logger.info(f"Metadatos cargados: {metadata}")
        return metadata
    
    def load_dataset(self):
        """
        Carga el dataset preparado para entrenamiento.
        
        Returns:
            tuple: Generadores para train/val/test y número de pasos.
        """
        # Cargar divisiones de datos
        splits_path = self.processed_data_path / 'pose_data' / 'splits.json'
        
        if not splits_path.exists():
            self.logger.error(f"No se encontró el archivo de splits: {splits_path}")
            raise FileNotFoundError(f"No se encontró el archivo: {splits_path}")
        
        with open(splits_path, 'r') as f:
            splits = json.load(f)
            
        self.logger.info(f"Cargadas divisiones de datos: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        
        # Función para crear data augmentation
        def create_augmentation_layers():
            """Crea capas de data augmentation."""
            return tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
                tf.keras.layers.RandomBrightness(0.1),
            ])
        
        # Función para cargar una imagen y sus keypoints
        def load_image_and_keypoints(img_id, split):
            """
            Carga una imagen y sus keypoints correspondientes.
            
            Args:
                img_id (str): ID de la imagen.
                split (str): División ('train', 'val', 'test').
                
            Returns:
                tuple: Imagen y keypoints.
            """
            # Cargar imagen
            img_path = self.processed_data_path / 'pose_data' / split / f"{img_id}.jpg"
            keypoints_path = self.processed_data_path / 'pose_data' / split / f"{img_id}_keypoints.json"
            
            try:
                # Leer imagen
                img = tf.io.read_file(str(img_path))
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [self.img_size, self.img_size])
                img = tf.cast(img, tf.float32) / 255.0  # Normalizar
                
                # Leer keypoints
                with open(keypoints_path, 'r') as f:
                    keypoints_data = json.load(f)
                
                # Extraer keypoints
                keypoints = np.array(keypoints_data['keypoints']).reshape(-1, 3)  # x, y, visibility
                
                # Normalizar coordenadas al rango [0,1]
                keypoints[:, 0] = keypoints[:, 0] / keypoints_data['width']
                keypoints[:, 1] = keypoints[:, 1] / keypoints_data['height']
                
                # Aplanar keypoints (solo x, y) para la salida
                keypoints_flat = keypoints[:, :2].reshape(-1)
                
                return img, keypoints_flat
            except Exception as e:
                self.logger.error(f"Error al cargar {img_id}: {e}")
                # Retornar valores por defecto en caso de error
                return tf.zeros([self.img_size, self.img_size, 3]), tf.zeros([self.num_keypoints * 2])
        
        # Crear datasets
        def create_dataset(split, is_training=False):
            """
            Crea un dataset TensorFlow para un split específico.
            
            Args:
                split (str): División ('train', 'val', 'test').
                is_training (bool): Si es True, aplica data augmentation.
                
            Returns:
                tf.data.Dataset: Dataset listo para entrenamiento.
            """
            # Lista de IDs para este split
            ids = splits[split]
            
            # Crear dataset de IDs
            ds = tf.data.Dataset.from_tensor_slices(ids)
            
            # Mapear para cargar imágenes y keypoints
            ds = ds.map(
                lambda img_id: tf.py_function(
                    func=lambda x: load_image_and_keypoints(x.numpy().decode('utf-8'), split),
                    inp=[img_id],
                    Tout=[tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Aplicar data augmentation solo a training
            if is_training and self.use_augmentation:
                aug_layers = create_augmentation_layers()
                
                # Aplicar augmentation a las imágenes (no a los keypoints)
                ds = ds.map(
                    lambda img, kp: (aug_layers(img, training=True), kp),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
            # Configurar dataset para entrenamiento
            ds = ds.shuffle(buffer_size=len(ids)) if is_training else ds
            ds = ds.batch(self.batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            
            return ds
        
        # Crear datasets para cada split
        train_ds = create_dataset('train', is_training=True)
        val_ds = create_dataset('val')
        test_ds = create_dataset('test')
        
        # Calcular pasos por epoch
        train_steps = len(splits['train']) // self.batch_size
        val_steps = len(splits['val']) // self.batch_size
        test_steps = len(splits['test']) // self.batch_size
        
        return train_ds, val_ds, test_ds, train_steps, val_steps, test_steps
    
    def build_model(self):
        """
        Construye el modelo de detección de poses basado en MobileNetV2.
        
        Returns:
            tf.keras.Model: Modelo compilado.
        """
        # Modelo base pre-entrenado
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Congelar algunas capas iniciales del modelo base
        # Esto es útil para fine-tuning
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        # Construcción del modelo
        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))
        x = base_model(inputs)
        
        # Capas adicionales para regresión de landmarks
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Capa de salida: 2 valores (x, y) por cada keypoint
        outputs = Dense(self.num_keypoints * 2, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compilar modelo
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Error cuadrático medio para regresión
            metrics=['mae']  # Error absoluto medio
        )
        
        self.logger.info(f"Modelo construido con {model.count_params()} parámetros")
        return model
    
    def create_callbacks(self):
        """
        Crea callbacks para el entrenamiento.
        
        Returns:
            list: Lista de callbacks.
        """
        # Directorio para checkpoints
        checkpoint_dir = self.model_save_path / 'checkpoints' / self.timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Guardar el mejor modelo
            ModelCheckpoint(
                filepath=str(checkpoint_dir / 'model-{epoch:02d}-{val_loss:.4f}.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            ),
            # Detener el entrenamiento si no hay mejora
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reducir learning rate si no hay mejora
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # Logs para TensorBoard
            TensorBoard(
                log_dir=str(self.log_path / self.timestamp),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        return callbacks
    
    def train(self, resume_from=None):
        """
        Entrena el modelo de detección de poses.
        
        Args:
            resume_from (str, optional): Ruta a un modelo guardado para continuar el entrenamiento.
        
        Returns:
            tuple: Historial de entrenamiento y modelo entrenado.
        """
        self.logger.info("Iniciando entrenamiento del modelo de poses...")
        
        # Cargar dataset
        train_ds, val_ds, test_ds, train_steps, val_steps, test_steps = self.load_dataset()
        
        # Construir o cargar modelo
        if resume_from:
            self.logger.info(f"Cargando modelo desde {resume_from} para continuar entrenamiento")
            model = load_model(resume_from)
        else:
            self.logger.info("Construyendo nuevo modelo")
            model = self.build_model()
        
        # Imprimir resumen del modelo
        model.summary(print_fn=lambda x: self.logger.info(x))
        
        # Crear callbacks
        callbacks = self.create_callbacks()
        
        # Entrenar modelo
        self.logger.info(f"Comenzando entrenamiento por {self.epochs} epochs")
        history = model.fit(
            train_ds,
            epochs=self.epochs,
            steps_per_epoch=train_steps,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar en conjunto de test
        self.logger.info("Evaluando modelo en conjunto de test")
        test_results = model.evaluate(test_ds, steps=test_steps)
        test_metrics = dict(zip(model.metrics_names, test_results))
        
        self.logger.info(f"Resultados en test: {test_metrics}")
        
        # Guardar modelo final
        final_model_path = self.model_save_path / f'pose_model_{self.timestamp}_final.h5'
        model.save(final_model_path)
        self.logger.info(f"Modelo guardado en {final_model_path}")
        
        # Crear copia con nombre estándar (para facilitar carga)
        standard_model_path = self.model_save_path / 'pose_model_final.h5'
        model.save(standard_model_path)
        
        # Guardar resultados y configuración
        results = {
            'timestamp': self.timestamp,
            'test_metrics': test_metrics,
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'mae': [float(x) for x in history.history['mae']],
                'val_mae': [float(x) for x in history.history['val_mae']]
            },
            'config': {
                'img_size': self.img_size,
                'num_keypoints': self.num_keypoints,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'use_augmentation': self.use_augmentation
            }
        }
        
        results_path = self.model_save_path / f'training_results_{self.timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Resultados guardados en {results_path}")
        
        return history, model, results
    
    def export_model(self, model=None, model_path=None):
        """
        Exporta el modelo entrenado a diferentes formatos (TFLite, SavedModel, .pb).
        
        Args:
            model (tf.keras.Model, optional): Modelo a exportar. Si es None, se carga desde model_path.
            model_path (str, optional): Ruta al modelo guardado. Si es None, se usa pose_model_final.h5.
        """
        self.logger.info("Exportando modelo a diferentes formatos...")
        
        # Cargar modelo si no se proporciona
        if model is None:
            if model_path is None:
                model_path = self.model_save_path / 'pose_model_final.h5'
            
            self.logger.info(f"Cargando modelo desde {model_path}")
            model = load_model(model_path)
        
        # 1. Exportar a TensorFlow Lite
        tflite_path = self.model_save_path / 'pose_model.tflite'
        self.logger.info(f"Exportando a TFLite: {tflite_path}")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        # 2. Exportar a SavedModel
        saved_model_path = self.model_save_path / 'saved_model'
        self.logger.info(f"Exportando a SavedModel: {saved_model_path}")
        
        tf.saved_model.save(model, str(saved_model_path))
        
        # 3. Exportar a formato .pb
        pb_path = self.model_save_path / 'pose_estimation_model.pb'
        self.logger.info(f"Exportando a .pb: {pb_path}")
        
        # Utilizar la versión compatible para exportar a .pb
        try:
            converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(
                str(saved_model_path)
            )
            pb_model = converter.convert()
            
            with open(pb_path, 'wb') as f:
                f.write(pb_model)
        except Exception as e:
            self.logger.error(f"Error al exportar a .pb: {e}")
            
            # Método alternativo para generar el .pb
            self.logger.info("Intentando método alternativo para generar .pb")
            try:
                # Guardar el grafo de operaciones
                from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
                
                # Obtener grafo concreto
                full_model = tf.function(lambda x: model(x))
                full_model = full_model.get_concrete_function(
                    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
                
                # Obtener grafo congelado
                frozen_func = convert_variables_to_constants_v2(full_model)
                frozen_func.graph.as_graph_def()
                
                # Guardar modelo congelado
                tf.io.write_graph(frozen_func.graph, str(self.model_save_path), 
                                  'pose_estimation_model.pb', as_text=False)
                
                self.logger.info(f"Modelo .pb generado exitosamente en {pb_path}")
            except Exception as e2:
                self.logger.error(f"Error en método alternativo para .pb: {e2}")
        
        # 4. Crear archivo de configuración para el modelo
        config_path = self.model_save_path / 'pose_config.json'
        config = {
            'model_type': 'MobileNetV2',
            'input_size': self.img_size,
            'num_keypoints': self.num_keypoints,
            'keypoints_format': 'x1,y1,x2,y2,...',
            'input_normalization': 'divide_by_255',
            'output_normalization': 'sigmoid',
            'training_date': self.timestamp,
            'formats_available': {
                'h5': str(standard_model_path.name),
                'tflite': str(tflite_path.name),
                'saved_model': 'saved_model',
                'pb': str(pb_path.name)
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        self.logger.info(f"Configuración del modelo guardada en {config_path}")
        self.logger.info("Exportación de modelo completada")
    
    def visualize_predictions(self, model=None, num_samples=5):
        """
        Visualiza las predicciones del modelo en muestras aleatorias.
        
        Args:
            model (tf.keras.Model, optional): Modelo a usar para predicción.
            num_samples (int): Número de muestras a visualizar.
            
        Returns:
            matplotlib.figure.Figure: Figura con las visualizaciones.
        """
        self.logger.info(f"Visualizando predicciones en {num_samples} muestras...")
        
        # Cargar modelo si no se proporciona
        if model is None:
            model_path = self.model_save_path / 'pose_model_final.h5'
            if not model_path.exists():
                self.logger.error(f"No se encontró el modelo: {model_path}")
                return None
                
            model = load_model(model_path)
        
        # Cargar dataset de test
        _, _, test_ds, _, _, _ = self.load_dataset()
        
        # Tomar algunas muestras aleatorias
        samples = []
        for images, keypoints in test_ds.take(1):
            # Seleccionar muestras aleatorias del batch
            indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
            samples = [(images[i].numpy(), keypoints[i].numpy()) for i in indices]
        
        # Hacer predicciones
        fig, axs = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))
        
        for i, (image, true_keypoints) in enumerate(samples):
            # Predicción
            pred_keypoints = model.predict(np.expand_dims(image, axis=0))[0]
            
            # Visualizar imagen original con keypoints reales
            ax1 = axs[i, 0] if len(samples) > 1 else axs[0]
            ax1.imshow(image)
            ax1.set_title('Keypoints reales')
            
            # Dibujar keypoints reales
            for j in range(self.num_keypoints):
                x = true_keypoints[j*2] * self.img_size
                y = true_keypoints[j*2+1] * self.img_size
                ax1.plot(x, y, 'ro', markersize=5)
            
            # Visualizar imagen con keypoints predichos
            ax2 = axs[i, 1] if len(samples) > 1 else axs[1]
            ax2.imshow(image)
            ax2.set_title('Keypoints predichos')
            
            # Dibujar keypoints predichos
            for j in range(self.num_keypoints):
                x = pred_keypoints[j*2] * self.img_size
                y = pred_keypoints[j*2+1] * self.img_size
                ax2.plot(x, y, 'go', markersize=5)
        
        plt.tight_layout()
        
        # Guardar la visualización
        vis_path = self.model_save_path / f'predictions_visualization_{self.timestamp}.png'
        plt.savefig(vis_path)
        self.logger.info(f"Visualización guardada en {vis_path}")
        
        return fig

if __name__ == "__main__":
    # Configurar logging básico si se ejecuta como script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Entrenar modelo
    trainer = PoseModelTrainer()
    history, model, results = trainer.train()
    
    # Exportar modelo
    trainer.export_model(model)
    
    # Visualizar algunas predicciones
    trainer.visualize_predictions(model)