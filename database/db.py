"""
Módulo para la gestión de la base de datos del sistema de prueba virtual de ropa.
Define la conexión a la base de datos y los modelos de datos utilizados en la aplicación.
"""

import sqlite3
import json
import os
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import logging


class Database:
    """
    Clase para gestionar la conexión a la base de datos.
    Implementa un patrón Singleton para asegurar una única instancia de conexión.
    """
    _instance = None
    
    def __new__(cls, db_path: str = 'data/virtual_fitting.db'):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance.db_path = db_path
            cls._instance.connection = None
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """
        Inicializa la base de datos, creando las tablas si no existen.
        """
        try:
            # Asegurar que el directorio para la BD existe
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Establecer conexión
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Para obtener resultados como diccionarios
            
            # Crear tablas si no existen
            self._create_tables()
            
            self.logger.info(f"Base de datos inicializada en {self.db_path}")
        except Exception as e:
            self.logger.error(f"Error al inicializar la base de datos: {str(e)}")
            raise
    
    def _create_tables(self) -> None:
        """
        Crea las tablas necesarias en la base de datos si no existen.
        """
        cursor = self.connection.cursor()
        
        # Tabla de usuarios
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password_hash TEXT,
            name TEXT,
            gender TEXT,
            birth_date TEXT,
            height REAL,
            weight REAL,
            created_at TEXT,
            last_login TEXT
        )
        ''')
        
        # Tabla de medidas corporales
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            height REAL,
            weight REAL,
            chest REAL,
            waist REAL,
            hip REAL,
            shoulder_width REAL,
            arm_length REAL,
            inseam REAL,
            neck REAL,
            thigh REAL,
            raw_data TEXT,  -- JSON con datos completos
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Tabla de prendas de ropa
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS clothing_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,  -- "shirt", "pants", etc.
            brand TEXT,
            name TEXT,
            size TEXT,
            color TEXT,
            image_path TEXT,
            metadata TEXT  -- JSON con datos adicionales
        )
        ''')
        
        # Tabla de pruebas virtuales
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS virtual_fittings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            measurement_id INTEGER,
            date TEXT,
            result_image_path TEXT,
            clothing_items TEXT,  -- JSON con IDs de prendas
            fit_scores TEXT,  -- JSON con puntuaciones de ajuste
            comments TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (measurement_id) REFERENCES measurements (id)
        )
        ''')
        
        # Tabla de tallas recomendadas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS size_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            measurement_id INTEGER,
            clothing_type TEXT,
            brand TEXT,
            recommended_size TEXT,
            fit_score REAL,
            details TEXT,  -- JSON con detalles
            date TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (measurement_id) REFERENCES measurements (id)
        )
        ''')
        
        self.connection.commit()
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Obtiene la conexión a la base de datos.
        
        Returns:
            Objeto de conexión a SQLite
        """
        if self.connection is None:
            self._initialize()
        return self.connection
    
    def execute_query(self, query: str, parameters: tuple = ()) -> List[Dict]:
        """
        Ejecuta una consulta SELECT y devuelve los resultados.
        
        Args:
            query: Consulta SQL
            parameters: Parámetros para la consulta
            
        Returns:
            Lista de filas como diccionarios
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, parameters)
            results = [dict(row) for row in cursor.fetchall()]
            return results
        except Exception as e:
            self.logger.error(f"Error al ejecutar consulta: {str(e)}")
            self.logger.error(f"Query: {query}, Params: {parameters}")
            raise
    
    def execute_insert(self, query: str, parameters: tuple = ()) -> int:
        """
        Ejecuta una consulta INSERT y devuelve el ID generado.
        
        Args:
            query: Consulta SQL
            parameters: Parámetros para la consulta
            
        Returns:
            ID generado por la operación
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, parameters)
            self.connection.commit()
            return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error al ejecutar inserción: {str(e)}")
            self.logger.error(f"Query: {query}, Params: {parameters}")
            self.connection.rollback()
            raise
    
    def execute_update(self, query: str, parameters: tuple = ()) -> int:
        """
        Ejecuta una consulta UPDATE o DELETE y devuelve el número de filas afectadas.
        
        Args:
            query: Consulta SQL
            parameters: Parámetros para la consulta
            
        Returns:
            Número de filas afectadas
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, parameters)
            self.connection.commit()
            return cursor.rowcount
        except Exception as e:
            self.logger.error(f"Error al ejecutar actualización: {str(e)}")
            self.logger.error(f"Query: {query}, Params: {parameters}")
            self.connection.rollback()
            raise
    
    def close(self) -> None:
        """
        Cierra la conexión a la base de datos.
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("Conexión a la base de datos cerrada")


class BaseModel:
    """
    Clase base para todos los modelos de datos.
    """
    
    def __init__(self):
        self.id = None
    
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> 'BaseModel':
        """
        Crea una instancia del modelo a partir de una fila de la base de datos.
        
        Args:
            row: Fila de la base de datos como diccionario
            
        Returns:
            Instancia del modelo
        """
        instance = cls()
        for key, value in row.items():
            setattr(instance, key, value)
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el modelo a un diccionario.
        
        Returns:
            Diccionario con los atributos del modelo
        """
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_') and not callable(value)}


class User(BaseModel):
    """
    Modelo que representa a un usuario del sistema.
    """
    
    def __init__(self, email: str = None, name: str = None, gender: str = None,
                 birth_date: str = None, height: float = None, weight: float = None):
        """
        Inicializa un nuevo usuario.
        
        Args:
            email: Correo electrónico
            name: Nombre completo
            gender: Género (male, female, etc)
            birth_date: Fecha de nacimiento
            height: Altura en cm
            weight: Peso en kg
        """
        super().__init__()
        self.email = email
        self.password_hash = None
        self.name = name
        self.gender = gender
        self.birth_date = birth_date
        self.height = height
        self.weight = weight
        self.created_at = datetime.datetime.now().isoformat()
        self.last_login = None
    
    def set_password(self, password: str) -> None:
        """
        Establece la contraseña del usuario (hash).
        En una implementación real usaríamos bcrypt o similar.
        
        Args:
            password: Contraseña en texto plano
        """
        # Simulación simple de hash (NO usar en producción)
        import hashlib
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    def check_password(self, password: str) -> bool:
        """
        Verifica si la contraseña es correcta.
        
        Args:
            password: Contraseña a verificar
            
        Returns:
            True si la contraseña es correcta
        """
        import hashlib
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()
    
    def update_last_login(self) -> None:
        """
        Actualiza la fecha del último inicio de sesión.
        """
        self.last_login = datetime.datetime.now().isoformat()


class Measurement(BaseModel):
    """
    Modelo que representa las medidas corporales de un usuario.
    """
    
    def __init__(self, user_id: int = None):
        """
        Inicializa nuevas medidas corporales.
        
        Args:
            user_id: ID del usuario asociado
        """
        super().__init__()
        self.user_id = user_id
        self.date = datetime.datetime.now().isoformat()
        self.height = None
        self.weight = None
        self.chest = None
        self.waist = None
        self.hip = None
        self.shoulder_width = None
        self.arm_length = None
        self.inseam = None
        self.neck = None
        self.thigh = None
        self.raw_data = "{}"  # JSON con todos los datos
    
    def from_landmarks(self, landmarks: Dict[str, Dict[str, float]], measurements: Dict[str, float]) -> None:
        """
        Actualiza las medidas a partir de landmarks y cálculos.
        
        Args:
            landmarks: Diccionario con landmarks detectados
            measurements: Diccionario con medidas calculadas
        """
        for key, value in measurements.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Guardar datos completos en raw_data
        data = {
            "landmarks": landmarks,
            "measurements": measurements
        }
        self.raw_data = json.dumps(data)
    
    def get_raw_data(self) -> Dict:
        """
        Obtiene los datos crudos como diccionario.
        
        Returns:
            Diccionario con los datos crudos
        """
        try:
            return json.loads(self.raw_data)
        except:
            return {}


class ClothingItem(BaseModel):
    """
    Modelo que representa una prenda de ropa.
    """
    
    def __init__(self, type: str = None, brand: str = None, name: str = None,
                 size: str = None, color: str = None, image_path: str = None):
        """
        Inicializa una nueva prenda.
        
        Args:
            type: Tipo de prenda (shirt, pants, etc)
            brand: Marca
            name: Nombre del modelo
            size: Talla
            color: Color
            image_path: Ruta a la imagen
        """
        super().__init__()
        self.type = type
        self.brand = brand
        self.name = name
        self.size = size
        self.color = color
        self.image_path = image_path
        self.metadata = "{}"  # JSON con datos adicionales
    
    def set_metadata(self, data: Dict) -> None:
        """
        Establece los metadatos de la prenda.
        
        Args:
            data: Diccionario con metadatos
        """
        self.metadata = json.dumps(data)
    
    def get_metadata(self) -> Dict:
        """
        Obtiene los metadatos como diccionario.
        
        Returns:
            Diccionario con metadatos
        """
        try:
            return json.loads(self.metadata)
        except:
            return {}


class VirtualFitting(BaseModel):
    """
    Modelo que representa una prueba virtual de ropa.
    """
    
    def __init__(self, user_id: int = None, measurement_id: int = None):
        """
        Inicializa una nueva prueba virtual.
        
        Args:
            user_id: ID del usuario
            measurement_id: ID de las medidas usadas
        """
        super().__init__()
        self.user_id = user_id
        self.measurement_id = measurement_id
        self.date = datetime.datetime.now().isoformat()
        self.result_image_path = None
        self.clothing_items = "[]"  # JSON con IDs de prendas
        self.fit_scores = "{}"  # JSON con puntuaciones de ajuste
        self.comments = None
    
    def set_clothing_items(self, items: List[int]) -> None:
        """
        Establece las prendas utilizadas en la prueba.
        
        Args:
            items: Lista de IDs de prendas
        """
        self.clothing_items = json.dumps(items)
    
    def get_clothing_items(self) -> List[int]:
        """
        Obtiene los IDs de las prendas.
        
        Returns:
            Lista de IDs de prendas
        """
        try:
            return json.loads(self.clothing_items)
        except:
            return []
    
    def set_fit_scores(self, scores: Dict[str, float]) -> None:
        """
        Establece las puntuaciones de ajuste.
        
        Args:
            scores: Diccionario con puntuaciones
        """
        self.fit_scores = json.dumps(scores)
    
    def get_fit_scores(self) -> Dict[str, float]:
        """
        Obtiene las puntuaciones de ajuste.
        
        Returns:
            Diccionario con puntuaciones
        """
        try:
            return json.loads(self.fit_scores)
        except:
            return {}


class SizeRecommendation(BaseModel):
    """
    Modelo que representa una recomendación de talla.
    """
    
    def __init__(self, user_id: int = None, measurement_id: int = None,
                 clothing_type: str = None, brand: str = None):
        """
        Inicializa una nueva recomendación de talla.
        
        Args:
            user_id: ID del usuario
            measurement_id: ID de las medidas usadas
            clothing_type: Tipo de prenda
            brand: Marca
        """
        super().__init__()
        self.user_id = user_id
        self.measurement_id = measurement_id
        self.clothing_type = clothing_type
        self.brand = brand
        self.recommended_size = None
        self.fit_score = None
        self.details = "{}"  # JSON con detalles
        self.date = datetime.datetime.now().isoformat()
    
    def set_details(self, details: Dict) -> None:
        """
        Establece los detalles de la recomendación.
        
        Args:
            details: Diccionario con detalles
        """
        self.details = json.dumps(details)
    
    def get_details(self) -> Dict:
        """
        Obtiene los detalles como diccionario.
        
        Returns:
            Diccionario con detalles
        """
        try:
            return json.loads(self.details)
        except:
            return {}