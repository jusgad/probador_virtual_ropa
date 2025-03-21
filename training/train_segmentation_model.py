"""
Módulo para entrenar el modelo de segmentación de ropa.

Este módulo implementa la funcionalidad para entrenar un modelo
de deep learning que segmenta las prendas de ropa en imágenes.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, Concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

class SegmentationModelTrainer:
    """
    Clase para entrenar el modelo de segmentación de ropa.
    """
    
    def __init__(self, config_path='config.json'):
        """
        Inicializa el entrenador del modelo de segmentación.
        
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
        self.train_config = self.config['training']['segmentation']
        self.batch_size = self.train_config.get('batch_size', 8)
        self.epochs = self.train_config.get('epochs', 50)
        self.img_size = self.train_config.get('img_size', 256)
        self.num_classes = self.train_config.get('num_classes', 3)  # Background, Persona, Ropa
        self.learning_rate = self.train_config.get('learning_rate', 0.0001)
        self.use_augmentation = self.train_config.get('augmentation', True)
        
        # Crear directorios necesarios
        self.model_save_path = self.model_save_path / 'segmentation'
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.log_path = self.log_path / 'segmentation'
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Timestamp para identificar esta sesión de entrenamiento
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.logger.info(f"Inicializado entrenador de modelo de segmentación con tamaño de imagen {self.img_size}x{self.img_size}")
    
    def load_metadata(self):
        """
        Carga los metadatos del dataset preparado.
        
        Returns:
            dict: Metadatos del dataset de segmentación.
        """
        metadata_path = self.processed_data_path / 'segmentation_data' / 'metadata.json'
        
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
        splits_path = self.processed_data_path / 'segmentation_data' / 'splits.json'
        
        if not splits_path.exists():
            self.logger.error(f"No se encontró el archivo de splits: {splits_path}")
            raise FileNotFoundError(f"No se encontró el archivo: {splits_path}")
        
        with open(splits_path, 'r') as f:
            splits = json.load(f)
            
        self.logger.info(f"Cargadas divisiones de datos: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        
        # Función para crear data augmentation
        def create_augmentation_layers():
            """Crea capas de data augmentation apropiadas para segmentación."""
            return tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ])
        
        # Función para cargar una imagen y su máscara
        def load_image_and_mask(img_file, split):
            """
            Carga una imagen y su máscara correspondiente.
            
            Args:
                img_file (str): Nombre del archivo de imagen.
                split (str): División ('train', 'val', 'test').
                
            Returns:
                tuple: Imagen y máscara.
            """
            # Rutas de imagen y máscara
            img_path = self.processed_data_path / 'segmentation_data' / split / 'images' / img_file
            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = self.processed_data_path / 'segmentation_data' / split / 'masks' / mask_file
            
            try:
                # Leer imagen
                img = tf.io.read_file(str(img_path))
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                img = tf.image.resize(img, [self.img_size, self.img_size])
                img = tf.cast(img, tf.float32) / 255.0  # Normalizar
                
                # Leer máscara
                mask = tf.io.read_file(str(mask_path))
                mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
                mask = tf.image.resize(mask, [self.img_size, self.img_size], method='nearest')
                
                # Convertir a one-hot encoding
                mask = tf.cast(mask, tf.int32)
                mask = tf.one_hot(tf.squeeze(mask), self.num_classes)
                
                return img, mask
            except Exception as e:
                self.logger.error(f"Error al cargar {img_file}: {e}")
                # Retornar valores por defecto en caso de error
                return (tf.zeros([self.img_size, self.img_size, 3]), 
                        tf.zeros([self.img_size, self.img_size, self.num_classes]))
        
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
            # Lista de archivos para este split
            files = splits[split]
            
            # Crear dataset de nombres de archivos
            ds = tf.data.Dataset.from_tensor_slices(files)
            
            # Mapear para cargar imágenes y máscaras
            ds = ds.map(
                lambda img_file: tf.py_function(
                    func=lambda x: load_image_and_mask(x.numpy().decode('utf-8'), split),
                    inp=[img_file],
                    Tout=[tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Aplicar data augmentation solo a training
            if is_training and self.use_augmentation:
                aug_layers = create_augmentation_layers()
                
                # Para segmentación, debemos aplicar la misma transformación a la imagen y la máscara
                def apply_augmentation(image, mask):
                    # Concatenar imagen y máscara para augmentation conjunta
                    concatenated = tf.concat([image, mask], axis=-1)
                    augmented = aug_layers(concatenated)
                    
                    # Separar nuevamente
                    augmented_image = augmented[..., :3]
                    augmented_mask = augmented[..., 3:]
                    
                    return augmented_image, augmented_mask
                
                ds = ds.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
            
            # Configurar dataset para entrenamiento
            ds = ds.shuffle(buffer_size=len(files)) if is_training else ds
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
    
    def build_unet_model(self):
        """
        Construye un modelo U-Net basado en MobileNetV2 para segmentación.
        
        Returns:
            tf.keras.Model: Modelo compilado.
        """
        # Encoder: MobileNetV2 como backbone
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Extraer capas específicas para crear skip connections
        layer_names = [
            'block_1_expand_relu',   # 128x128
            'block_3_expand_relu',   # 64x64
            'block_6_expand_relu',   # 32x32
            'block_13_expand_relu',  # 16x16
            'block_16_project',      # 8x8
        ]
        
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
        
        # Crear modelo del encoder
        encoder = Model(inputs=base_model.input, outputs=base_model_outputs)
        encoder.trainable = True
        
        # Congelar algunas capas iniciales del encoder
        for layer in encoder.layers[:100]:
            layer.trainable = False
        
        # Entrada del modelo
        inputs = Input(shape=(self.img_size, self.img_size, 3))
        
        # Encoder outputs (features en diferentes escalas)
        skips = encoder(inputs)
        x = skips[-1]  # feature map más pequeño
        
        # Decoder blocks con skip connections
        for i in range(len(skips)-2, -1, -1):
            # Calcular el factor de escala
            if i >= 3:  # Para las capas más pequeñas
                up_size = (2, 2)
            else:  # Para las capas más grandes
                up_size = (2, 2)
            
            # Upsampling
            x = Conv2DTranspose(
                filters=256 if i > 2 else 128,
                kernel_size=3,
                strides=2,
                padding='same',
                use_bias=False
            )(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            # Skip connection
            concat = Concatenate()([x, skips[i]])
            
            # Refinar features
            x = Conv2D(
                filters=256 if i > 2 else 128,
                kernel_size=3,
                padding='same',
                use_bias=False
            )(concat)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            # Segunda convolución
            x = Conv2D(
                filters=256 if i > 2 else 128,
                kernel_size=3,
                padding='same',
                use_bias=False
            )(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            # Dropout para regularización
            if i > 1:
                x = Dropout(0.3)(x)
        
        # Capa final para generar la segmentación
        outputs = Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            padding='same',
            activation='softmax'
        )(x)
        
        # Crear modelo
        model = Model(inputs=inputs, outputs=outputs)
        
        # Definir pérdida y métricas
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [
            'accuracy',
            tf.keras.metrics.MeanIoU(num_classes=self.num_classes)
        ]
        
        # Compilar modelo
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"Modelo U-Net construido con {model.count_params()} parámetros")
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
                filepath=str(checkpoint_dir / 'model-{epoch:02d}-{val_mean_io_u:.4f}.h5'),
                monitor='val_mean_io_u',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            # Detener el entrenamiento si no hay mejora
            EarlyStopping(
                monitor='val_mean_io_u',
                patience=10,
                restore_best_weights=True,
                mode='max',
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
        Entrena el modelo de segmentación.
        
        Args:
            resume_from (str, optional): Ruta a un modelo guardado para continuar el entrenamiento.
        
        Returns:
            tuple: Historial de entrenamiento y modelo entrenado.
        """
        self.logger.info("Iniciando entrenamiento del modelo de segmentación...")
        
        # Cargar dataset
        train_ds, val_ds, test_ds, train_steps, val_steps, test_steps = self.load_dataset()
        
        # Construir o cargar modelo
        if resume_from:
            self.logger.info(f"Cargando modelo desde {resume_from} para continuar entrenamiento")
            model = load_model(resume_from)
        else:
            self.logger.info("Construyendo nuevo modelo")
            model = self.build_unet_model()
        
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
        final_model_path = self.model_save_path / f'segmentation_model_{self.timestamp}_final.h5'
        model.save(final_model_path)
        self.logger.info(f"Modelo guardado en {final_model_path}")
        
        # Crear copia con nombre estándar (para facilitar carga)
        standard_model_path = self.model_save_path / 'segmentation_model_final.h5'
        model.save(standard_model_path)
        
        # Guardar resultados y configuración
        results = {
            'timestamp': self.timestamp,
            'test_metrics': {k: float(v) if isinstance(v, np.float32) else v for k, v in test_metrics.items()},
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'mean_io_u': [float(x) for x in history.history['mean_io_u']],
                'val_mean_io_u': [float(x) for x in history.history['val_mean_io_u']]
            },
            'config': {
                'img_size': self.img_size,
                'num_classes': self.num_classes,
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
            model_path (str, optional): Ruta al modelo guardado. Si es None, se usa segmentation_model_final.h5.
        """
        self.logger.info("Exportando modelo a diferentes formatos...")
        
        # Cargar modelo si no se proporciona
        if model is None:
            if model_path is None:
                model_path = self.model_save_path / 'segmentation_model_final.h5'
            
            self.logger.info(f"Cargando modelo desde {model_path}")
            model = load_model(model_path)
        
        # 1. Exportar a TensorFlow Lite
        tflite_path = self.model_save_path / 'clothing_segmentation.tflite'
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
        pb_path = self.model_save_path / 'human_segmentation.pb'
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
                                  'human_segmentation.pb', as_text=False)
                
                self.logger.info(f"Modelo .pb generado exitosamente en {pb_path}")
            except Exception as e2:
                self.logger.error(f"Error en método alternativo para .pb: {e2}")
        
        # 4. Crear archivo de configuración para el modelo
        config_path = self.model_save_path / 'segmentation_config.json'
        class_mapping = {
            '0': 'background',
            '1': 'person',
            '2': 'clothing'
        }
        
        config = {
            'model_type': 'U-Net_MobileNetV2',
            'input_size': self.img_size,
            'num_classes': self.num_classes,
            'class_mapping': class_mapping,
            'input_normalization': 'divide_by_255',
            'output_type': 'softmax',
            'training_date': self.timestamp,
            'formats_available': {
                'h5': 'segmentation_model_final.h5',
                'tflite': 'clothing_segmentation.tflite',
                'saved_model': 'saved_model',
                'pb': 'human_segmentation.pb'
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
            model_path = self.model_save_path / 'segmentation_model_final.h5'
            if not model_path.exists():
                self.logger.error(f"No se encontró el modelo: {model_path}")
                return None
                
            model = load_model(model_path)
        
        # Cargar dataset de test
        _, _, test_ds, _, _, _ = self.load_dataset()
        
        # Tomar algunas muestras aleatorias
        samples = []
        for images, masks in test_ds.take(1):
            # Seleccionar muestras aleatorias del batch
            indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
            samples = [(images[i].numpy(), masks[i].numpy()) for i in indices]
        
        # Hacer predicciones
        fig, axs = plt.subplots(len(samples), 3, figsize=(15, 5 * len(samples)))
        
        # Para visualización: asignar colores a cada clase
        colors = np.array([
            [0, 0, 0],       # Fondo: negro
            [0, 0, 255],     # Persona: azul
            [255, 0, 0]      # Ropa: rojo
        ]) / 255.0
        
        for i, (image, true_mask) in enumerate(samples):
            # Predicción
            pred_mask = model.predict(np.expand_dims(image, axis=0))[0]
            
            # Convertir máscaras one-hot a índices de clase
            true_mask_idx = np.argmax(true_mask, axis=-1)
            pred_mask_idx = np.argmax(pred_mask, axis=-1)
            
            # Convertir índices a imagen RGB para visualización
            true_mask_rgb = colors[true_mask_idx]
            pred_mask_rgb = colors[pred_mask_idx]
            
            # Visualizar imagen original
            if len(samples) > 1:
                ax1 = axs[i, 0]
                ax2 = axs[i, 1]
                ax3 = axs[i, 2]
            else:
                ax1 = axs[0]
                ax2 = axs[1]
                ax3 = axs[2]
            
            ax1.imshow(image)
            ax1.set_title('Imagen original')
            ax1.axis('off')
            
            # Visualizar máscara real
            ax2.imshow(true_mask_rgb)
            ax2.set_title('Máscara real')
            ax2.axis('off')
            
            # Visualizar máscara predicha
            ax3.imshow(pred_mask_rgb)
            ax3.set_title('Máscara predicha')
            ax3.axis('off')
        
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
    trainer = SegmentationModelTrainer()
    history, model, results = trainer.train()
    
    # Exportar modelo
    trainer.export_model(model)
    
    # Visualizar algunas predicciones
    trainer.visualize_predictions(model)