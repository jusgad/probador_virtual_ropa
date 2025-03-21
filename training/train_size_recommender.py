"""
Módulo para entrenar el modelo de recomendación de tallas.

Este módulo implementa la funcionalidad para entrenar un modelo
que recomienda tallas de ropa basado en las medidas corporales.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import logging
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class SizeRecommenderTrainer:
    """
    Clase para entrenar el modelo de recomendación de tallas.
    """
    
    def __init__(self, config_path='config.json'):
        """
        Inicializa el entrenador del modelo de recomendación de tallas.
        
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
        self.train_config = self.config['training']['size_recommender']
        self.batch_size = self.train_config.get('batch_size', 32)
        self.epochs = self.train_config.get('epochs', 100)
        self.learning_rate = self.train_config.get('learning_rate', 0.001)
        self.early_stopping_patience = self.train_config.get('early_stopping_patience', 10)
        
        # Crear directorios necesarios
        self.model_save_path = self.model_save_path / 'size_recommender'
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.log_path = self.log_path / 'size_recommender'
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Timestamp para identificar esta sesión de entrenamiento
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.logger.info("Inicializado entrenador de modelo de recomendación de tallas")
    
    def load_data(self):
        """
        Carga y preprocesa los datos para el entrenamiento.
        
        Returns:
            tuple: Datos de entrenamiento, validación y prueba, junto con información de procesamiento.
        """
        self.logger.info("Cargando datos para entrenamiento")
        
        # Verificar existencia de archivos
        train_path = self.processed_data_path / 'size_data' / 'train.csv'
        val_path = self.processed_data_path / 'size_data' / 'val.csv'
        test_path = self.processed_data_path / 'size_data' / 'test.csv'
        
        for path in [train_path, val_path, test_path]:
            if not path.exists():
                self.logger.error(f"No se encontró el archivo: {path}")
                raise FileNotFoundError(f"No se encontró el archivo: {path}")
        
        # Cargar dataframes
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        self.logger.info(f"Cargados datos: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        # Cargar metadata para identificar columnas
        metadata_path = self.processed_data_path / 'size_data' / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Identificar columnas de características y objetivos
        feature_cols = metadata['feature_columns']
        target_cols = metadata['target_columns']
        
        self.logger.info(f"Características: {feature_cols}")
        self.logger.info(f"Objetivos: {target_cols}")
        
        # Extraer características y objetivos
        X_train = train_df[feature_cols].values
        y_train = {col: train_df[col].values for col in target_cols}
        
        X_val = val_df[feature_cols].values
        y_val = {col: val_df[col].values for col in target_cols}
        
        X_test = test_df[feature_cols].values
        y_test = {col: test_df[col].values for col in target_cols}
        
        # Normalizar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Guardar el scaler
        scaler_path = self.model_save_path / 'scaler.pkl'
        joblib.dump(scaler, scaler_path)
        self.logger.info(f"Scaler guardado en {scaler_path}")
        
        # Procesar variables categóricas (objetivos)
        encoders = {}
        y_train_encoded = {}
        y_val_encoded = {}
        y_test_encoded = {}
        label_maps = {}
        
        for col in target_cols:
            # Crear LabelEncoder
            le = LabelEncoder()
            
            # Ajustar y transformar datos de entrenamiento
            y_train_encoded[col] = le.fit_transform(y_train[col])
            
            # Transformar datos de validación y prueba
            y_val_encoded[col] = le.transform(y_val[col])
            y_test_encoded[col] = le.transform(y_test[col])
            
            # Guardar encoder
            encoders[col] = le
            
            # Guardar mapeo de etiquetas
            label_maps[col] = {i: label for i, label in enumerate(le.classes_)}
            
            self.logger.info(f"Clases para {col}: {le.classes_}")
        
        # Guardar los encoders
        encoders_path = self.model_save_path / 'encoders.pkl'
        joblib.dump(encoders, encoders_path)
        self.logger.info(f"Encoders guardados en {encoders_path}")
        
        # Guardar información de características y etiquetas
        feature_info = {
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'feature_means': scaler.mean_.tolist(),
            'feature_stds': scaler.scale_.tolist(),
            'label_maps': label_maps
        }
        
        feature_info_path = self.model_save_path / 'feature_info.json'
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        self.logger.info(f"Información de características guardada en {feature_info_path}")
        
        # Información adicional para la construcción del modelo
        target_dims = {col: len(encoders[col].classes_) for col in target_cols}
        
        return (
            X_train_scaled, y_train_encoded, 
            X_val_scaled, y_val_encoded, 
            X_test_scaled, y_test_encoded,
            feature_cols, target_cols, encoders, target_dims
        )
    
    def prepare_tensorflow_datasets(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Convierte los datos a datasets de TensorFlow.
        
        Args:
            X_train, y_train, X_val, y_val, X_test, y_test: Datos preprocesados.
            
        Returns:
            tuple: Datasets de TensorFlow para entrenamiento, validación y prueba.
        """
        # Convertir diccionarios de etiquetas a listas
        y_train_list = [y_train[k] for k in sorted(y_train.keys())]
        y_val_list = [y_val[k] for k in sorted(y_val.keys())]
        y_test_list = [y_test[k] for k in sorted(y_test.keys())]
        
        # Crear datasets de TensorFlow
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, tuple(y_train_list)))
        train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, tuple(y_val_list)))
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, tuple(y_test_list)))
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, test_ds
    
    def build_model(self, input_dim, target_dims):
        """
        Construye el modelo de recomendación de tallas.
        
        Args:
            input_dim (int): Dimensión de entrada (número de características).
            target_dims (dict): Diccionario con las dimensiones de salida para cada objetivo.
            
        Returns:
            tf.keras.Model: Modelo compilado.
        """
        self.logger.info(f"Construyendo modelo con dimensión de entrada {input_dim}")
        self.logger.info(f"Dimensiones de salida: {target_dims}")
        
        # Entrada del modelo
        inputs = Input(shape=(input_dim,))
        
        # Capas compartidas
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Crear salidas múltiples para diferentes tipos de tallas
        outputs = []
        losses = {}
        loss_weights = {}
        metrics = {}
        
        for i, (target_name, num_classes) in enumerate(sorted(target_dims.items())):
            # Capa específica para cada tipo de talla
            branch = Dense(32, activation='relu', name=f"{target_name}_branch")(x)
            output = Dense(num_classes, activation='softmax', name=target_name)(branch)
            outputs.append(output)
            
            # Configurar pérdida y métrica para esta salida
            losses[target_name] = 'sparse_categorical_crossentropy'
            loss_weights[target_name] = 1.0
            metrics[target_name] = ['accuracy']
        
        # Crear modelo multi-salida
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compilar modelo
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        self.logger.info(f"Modelo construido con {model.count_params()} parámetros")
        return model
    
    def create_callbacks(self, target_cols):
        """
        Crea callbacks para el entrenamiento.
        
        Args:
            target_cols (list): Lista de columnas objetivo.
            
        Returns:
            list: Lista de callbacks.
        """
        # Directorio para checkpoints
        checkpoint_dir = self.model_save_path / 'checkpoints' / self.timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Definir el monitor basado en el promedio de precisión de todas las salidas
        # Si solo hay una salida, usar esa directamente
        if len(target_cols) == 1:
            monitor = f"val_{target_cols[0]}_accuracy"
        else:
            # Para múltiples salidas, promediar las precisiones de validación
            monitor = "val_loss"
        
        callbacks = [
            # Guardar el mejor modelo
            ModelCheckpoint(
                filepath=str(checkpoint_dir / 'model-{epoch:02d}-{' + monitor + ':.4f}.h5'),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                verbose=1
            ),
            # Detener el entrenamiento si no hay mejora
            EarlyStopping(
                monitor=monitor,
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            # Reducir learning rate si no hay mejora
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=self.early_stopping_patience // 2,
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
        Entrena el modelo de recomendación de tallas.
        
        Args:
            resume_from (str, optional): Ruta a un modelo guardado para continuar el entrenamiento.
        
        Returns:
            tuple: Historial de entrenamiento, modelo entrenado y resultados.
        """
        self.logger.info("Iniciando entrenamiento del modelo de recomendación de tallas...")
        
        # Cargar y preprocesar datos
        (X_train, y_train, X_val, y_val, X_test, y_test,
         feature_cols, target_cols, encoders, target_dims) = self.load_data()
        
        input_dim = X_train.shape[1]
        
        # Crear datasets de TensorFlow
        train_ds, val_ds, test_ds = self.prepare_tensorflow_datasets(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Construir o cargar modelo
        if resume_from:
            self.logger.info(f"Cargando modelo desde {resume_from} para continuar entrenamiento")
            model = load_model(resume_from)
        else:
            self.logger.info("Construyendo nuevo modelo")
            model = self.build_model(input_dim, target_dims)
        
        # Imprimir resumen del modelo
        model.summary(print_fn=lambda x: self.logger.info(x))
        
        # Crear callbacks
        callbacks = self.create_callbacks(target_cols)
        
        # Entrenar modelo
        self.logger.info(f"Comenzando entrenamiento por {self.epochs} epochs")
        history = model.fit(
            train_ds,
            epochs=self.epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar en conjunto de test
        self.logger.info("Evaluando modelo en conjunto de test")
        test_results = model.evaluate(test_ds, verbose=1)
        
        # Crear un diccionario de métricas de test
        test_metrics = {}
        for i, metric_name in enumerate(model.metrics_names):
            test_metrics[metric_name] = float(test_results[i])
        
        self.logger.info(f"Resultados en test: {test_metrics}")
        
        # Guardar modelo final
        final_model_path = self.model_save_path / f'size_model_{self.timestamp}_final.h5'
        model.save(final_model_path)
        self.logger.info(f"Modelo guardado en {final_model_path}")
        
        # Crear copia con nombre estándar (para facilitar carga)
        standard_model_path = self.model_save_path / 'size_model_final.h5'
        model.save(standard_model_path)
        
        # Preparar historial para serialización
        history_dict = {}
        for key in history.history:
            history_dict[key] = [float(x) for x in history.history[key]]
        
        # Guardar resultados y configuración
        results = {
            'timestamp': self.timestamp,
            'test_metrics': test_metrics,
            'training_history': history_dict,
            'config': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'early_stopping_patience': self.early_stopping_patience,
                'input_dim': input_dim,
                'target_dims': target_dims
            },
            'data_info': {
                'num_train_samples': len(X_train),
                'num_val_samples': len(X_val),
                'num_test_samples': len(X_test),
                'feature_cols': feature_cols,
                'target_cols': target_cols
            }
        }
        
        results_path = self.model_save_path / f'training_results_{self.timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Resultados guardados en {results_path}")
        
        return history, model, results
    
    def evaluate_model(self, model=None, X_test=None, y_test=None, encoders=None, target_cols=None):
        """
        Evalúa el modelo con métricas detalladas y genera visualizaciones.
        
        Args:
            model (tf.keras.Model, optional): Modelo a evaluar.
            X_test, y_test: Datos de prueba.
            encoders: Codificadores para las etiquetas.
            target_cols: Nombres de las columnas objetivo.
            
        Returns:
            dict: Resultados de la evaluación.
        """
        self.logger.info("Evaluando modelo con métricas detalladas...")
        
        # Cargar modelo si no se proporciona
        if model is None:
            model_path = self.model_save_path / 'size_model_final.h5'
            if not model_path.exists():
                self.logger.error(f"No se encontró el modelo: {model_path}")
                return None
                
            model = load_model(model_path)
        
        # Cargar datos de test si no se proporcionan
        if X_test is None or y_test is None or encoders is None or target_cols is None:
            (_, _, _, _, X_test, y_test,
             _, target_cols, encoders, _) = self.load_data()
        
        # Hacer predicciones
        predictions = model.predict(X_test)
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Metrics directory for saving reports and visualizations
        metrics_dir = self.model_save_path / 'evaluation' / self.timestamp
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        evaluation_results = {}
        
        # Para cada objetivo, generar métricas detalladas
        for i, target_col in enumerate(sorted(target_cols)):
            # Convertir predicciones a índices de clase
            y_pred = np.argmax(predictions[i], axis=1)
            y_true = y_test[target_col]
            
            # Obtener nombres de clases
            class_names = encoders[target_col].classes_
            
            # Calcular métricas
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            
            self.logger.info(f"Resultados para {target_col}:")
            self.logger.info(f"Precisión: {report['accuracy']:.4f}")
            
            # Guardar reporte de clasificación
            report_path = metrics_dir / f"{target_col}_classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Visualizar matriz de confusión
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.title(f'Matriz de Confusión para {target_col}')
            
            # Guardar visualización
            cm_path = metrics_dir / f"{target_col}_confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            
            # Guardar resultados
            evaluation_results[target_col] = {
                'accuracy': float(report['accuracy']),
                'class_report': report,
                'confusion_matrix': cm.tolist()
            }
        
        # Guardar resultados completos
        eval_results_path = metrics_dir / 'evaluation_results.json'
        with open(eval_results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
            
        self.logger.info(f"Resultados de evaluación guardados en {eval_results_path}")
        
        return evaluation_results
    
    def visualize_training_history(self, history=None):
        """
        Visualiza el historial de entrenamiento.
        
        Args:
            history: Historial de entrenamiento.
            
        Returns:
            list: Lista de figuras generadas.
        """
        self.logger.info("Visualizando historial de entrenamiento...")
        
        # Si no se proporciona historial, intentar cargarlo desde archivos
        if history is None:
            # Buscar el archivo de resultados más reciente
            results_files = list(self.model_save_path.glob('training_results_*.json'))
            if not results_files:
                self.logger.error("No se encontraron archivos de resultados de entrenamiento")
                return None
                
            # Cargar el más reciente
            latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
            with open(latest_results, 'r') as f:
                results = json.load(f)
                
            history = results['training_history']
        elif hasattr(history, 'history'):
            # Si es un objeto History de Keras, obtener su diccionario
            history = history.history
        
        figures = []
        
        # Directorio para guardar visualizaciones
        vis_dir = self.model_save_path / 'visualizations' / self.timestamp
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Gráfico de pérdida
        loss_metrics = [key for key in history.keys() if 'loss' in key and not 'val' in key]
        val_loss_metrics = [key for key in history.keys() if 'loss' in key and 'val' in key]
        
        plt.figure(figsize=(12, 6))
        for metric in loss_metrics:
            plt.plot(history[metric], label=metric)
        for metric in val_loss_metrics:
            plt.plot(history[metric], label=metric, linestyle='--')
            
        plt.title('Evolución de la Pérdida')
        plt.xlabel('Epochs')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Guardar figura
        loss_path = vis_dir / 'loss_history.png'
        plt.savefig(loss_path)
        plt.close()
        figures.append(loss_path)
        
        # 2. Gráficos de precisión para cada objetivo
        acc_metrics = [key for key in history.keys() if 'accuracy' in key and not 'val' in key]
        val_acc_metrics = [key for key in history.keys() if 'accuracy' in key and 'val' in key]
        
        plt.figure(figsize=(12, 6))
        for metric in acc_metrics:
            plt.plot(history[metric], label=metric)
        for metric in val_acc_metrics:
            plt.plot(history[metric], label=metric, linestyle='--')
            
        plt.title('Evolución de la Precisión')
        plt.xlabel('Epochs')
        plt.ylabel('Precisión')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Guardar figura
        acc_path = vis_dir / 'accuracy_history.png'
        plt.savefig(acc_path)
        plt.close()
        figures.append(acc_path)
        
        self.logger.info(f"Visualizaciones guardadas en {vis_dir}")
        
        return figures
    
    def export_model_for_production(self, model=None):
        """
        Exporta el modelo para producción.
        
        Args:
            model (tf.keras.Model, optional): Modelo a exportar.
        """
        self.logger.info("Exportando modelo para producción...")
        
        # Cargar modelo si no se proporciona
        if model is None:
            model_path = self.model_save_path / 'size_model_final.h5'
            if not model_path.exists():
                self.logger.error(f"No se encontró el modelo: {model_path}")
                return
                
            model = load_model(model_path)
        
        # 1. SavedModel format
        saved_model_path = self.model_save_path / 'saved_model'
        self.logger.info(f"Exportando a SavedModel: {saved_model_path}")
        
        try:
            tf.saved_model.save(model, str(saved_model_path))
        except Exception as e:
            self.logger.error(f"Error al exportar a SavedModel: {e}")
        
        # 2. Crear un archivo de configuración
        # Incluir toda la información necesaria para usar el modelo
        
        # Cargar información de características
        feature_info_path = self.model_save_path / 'feature_info.json'
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
        
        # Configuración para inferencia
        inference_config = {
            'model_type': 'MultiOutputClassifier',
            'input_features': feature_info['feature_cols'],
            'target_outputs': feature_info['target_cols'],
            'input_normalization': {
                'means': feature_info['feature_means'],
                'stds': feature_info['feature_stds']
            },
            'output_labels': feature_info['label_maps'],
            'training_date': self.timestamp,
            'formats_available': {
                'h5': 'size_model_final.h5',
                'saved_model': 'saved_model'
            },
            'preprocessing': {
                'scaler': 'scaler.pkl',
                'encoders': 'encoders.pkl'
            }
        }
        
        # Guardar configuración
        inference_config_path = self.model_save_path / 'size_recommender_config.json'
        with open(inference_config_path, 'w') as f:
            json.dump(inference_config, f, indent=2)
            
        self.logger.info(f"Configuración para inferencia guardada en {inference_config_path}")
        self.logger.info("Exportación de modelo completada")

if __name__ == "__main__":
    # Configurar logging básico si se ejecuta como script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Entrenar modelo
    trainer = SizeRecommenderTrainer()
    history, model, results = trainer.train()
    
    # Evaluar modelo
    evaluation_results = trainer.evaluate_model(model)
    
    # Visualizar historial
    trainer.visualize_training_history(history)
    
    # Exportar modelo
    trainer.export_model_for_production(model)