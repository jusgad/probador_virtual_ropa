"""
Módulo para la gestión de la base de datos del sistema de prueba virtual de ropa.
Define la conexión a la base de datos y los modelos de datos utilizados en la aplicación.
"""

import sqlite3
import json
import os
import datetime
import threading
from typing import Dict, List, Any, Optional
import logging
from werkzeug.security import generate_password_hash, check_password_hash

# Objeto para almacenar la conexión por hilo (Thread-safety)
class Database:
    """
    Clase para gestionar la conexión a la base de datos.
    Implementa un patrón Singleton y utiliza thread-local storage para asegurar seguridad en multi-hilo.
    """
    _instance = None
    _local = threading.local()
    
    def __new__(cls, db_path: str = 'data/virtual_fitting.db'):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance.db_path = db_path
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance._init_db_file()
        return cls._instance
        
    def _init_db_file(self) -> None:
        """Inicializa el archivo de base de datos y crea las tablas si no existen."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            # Conexión temporal para crear tablas
            conn = sqlite3.connect(self.db_path)
            self._create_tables_with_conn(conn)
            conn.close()
            self.logger.info(f"Archivo de base de datos listo en {self.db_path}")
        except Exception as e:
            self.logger.error(f"Error al inicializar el archivo de base de datos: {str(e)}")
            raise

    @property
    def connection(self) -> sqlite3.Connection:
        """Obtiene o crea una conexión SQLite específica para el hilo actual."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def close_connection(self) -> None:
        """Cierra la conexión SQLite del hilo actual si está abierta."""
        if hasattr(self._local, 'connection') and self._local.connection is not None:
            try:
                self._local.connection.close()
            except Exception as e:
                self.logger.error(f"Error al cerrar conexión de hilo: {e}")
            finally:
                self._local.connection = None

    def _create_tables_with_conn(self, conn: sqlite3.Connection) -> None:
        """Crea las tablas necesarias si no existen."""
        cursor = conn.cursor()
        
        # Tabla de usuarios (esquema alineado con setup_db.py)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT,
            first_name TEXT,
            last_name TEXT,
            gender TEXT CHECK(gender IN ('male', 'female', 'other', 'not_specified')),
            birth_date TEXT,
            profile_image TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT
        )
        ''')
        
        # Tabla de preferencias de usuario
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            preferred_fit TEXT DEFAULT 'regular' CHECK(preferred_fit IN ('slim', 'regular', 'loose')),
            preferred_brands TEXT,
            preferred_categories TEXT,
            preferred_colors TEXT,
            size_region TEXT DEFAULT 'EU' CHECK(size_region IN ('EU', 'US', 'UK', 'INT')),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')

        # Tabla de marcas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS brands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            logo_url TEXT,
            website TEXT,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Tabla de tablas de tallas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS size_charts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_id INTEGER,
            gender TEXT CHECK(gender IN ('male', 'female', 'unisex')),
            category TEXT,
            chart_data TEXT, -- JSON
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (brand_id) REFERENCES brands(id) ON DELETE CASCADE,
            UNIQUE (brand_id, gender, category)
        )
        ''')

        # Tabla de prendas de ropa (clothing)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS clothing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand_id INTEGER,
            category TEXT NOT NULL,
            subcategory TEXT,
            type TEXT NOT NULL,
            gender TEXT CHECK(gender IN ('male', 'female', 'unisex')),
            description TEXT,
            color TEXT,
            material TEXT,
            price REAL,
            currency TEXT DEFAULT 'EUR',
            product_url TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (brand_id) REFERENCES brands(id) ON DELETE SET NULL
        )
        ''')

        # Tabla de imágenes de prendas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS clothing_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clothing_id INTEGER NOT NULL,
            image_url TEXT NOT NULL,
            image_type TEXT DEFAULT 'main' CHECK(image_type IN ('main', 'thumbnail', 'detail', 'back', 'side')),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (clothing_id) REFERENCES clothing(id) ON DELETE CASCADE
        )
        ''')

        # Tabla de tallas de prendas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS clothing_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clothing_id INTEGER NOT NULL,
            size_name TEXT NOT NULL,
            measurements TEXT NOT NULL, -- JSON
            stock INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (clothing_id) REFERENCES clothing(id) ON DELETE CASCADE,
            UNIQUE (clothing_id, size_name)
        )
        ''')

        # Tabla de medidas corporales (user_measurements)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT DEFAULT 'Mis medidas',
            height REAL,
            weight REAL,
            chest REAL,
            waist REAL,
            hips REAL,
            shoulders REAL,
            arm_length REAL,
            inseam REAL,
            neck REAL,
            thigh REAL,  -- Agregado para consistencia con app y repositorios
            additional_measurements TEXT,
            front_image TEXT,
            side_image TEXT,
            landmarks TEXT,
            is_current BOOLEAN DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')

        # Tabla de resultados de ajuste (fitting_results)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fitting_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            clothing_id INTEGER NOT NULL,
            size_name TEXT NOT NULL,
            measurement_id INTEGER NOT NULL,
            fit_score REAL NOT NULL,
            fit_type TEXT CHECK(fit_type IN ('tight', 'regular', 'loose')),
            fit_details TEXT, -- JSON
            preview_image TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (clothing_id) REFERENCES clothing(id) ON DELETE CASCADE,
            FOREIGN KEY (measurement_id) REFERENCES user_measurements(id) ON DELETE CASCADE
        )
        ''')

        # Tabla de recomendaciones de talla para compatibilidad con repositorios
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS size_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            measurement_id INTEGER,
            clothing_type TEXT,
            brand TEXT,
            recommended_size TEXT,
            fit_score REAL,
            details TEXT, -- JSON
            date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (measurement_id) REFERENCES user_measurements(id) ON DELETE CASCADE
        )
        ''')
        
        conn.commit()

    def get_connection(self) -> sqlite3.Connection:
        """Obtiene la conexión a la base de datos."""
        return self.connection

    def execute_query(self, query: str, parameters: tuple = ()) -> List[Dict]:
        """Ejecuta una consulta SELECT y devuelve los resultados."""
        try:
            with self.connection:
                cursor = self.connection.cursor()
                cursor.execute(query, parameters)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error al ejecutar consulta: {str(e)}")
            self.logger.error(f"Query: {query}, Params: {parameters}")
            raise

    def execute_insert(self, query: str, parameters: tuple = ()) -> int:
        """Ejecuta una consulta INSERT y devuelve el ID generado."""
        try:
            with self.connection:
                cursor = self.connection.cursor()
                cursor.execute(query, parameters)
                return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error al ejecutar inserción: {str(e)}")
            self.logger.error(f"Query: {query}, Params: {parameters}")
            raise

    def execute_update(self, query: str, parameters: tuple = ()) -> int:
        """Ejecuta una consulta UPDATE o DELETE y devuelve el número de filas afectadas."""
        try:
            with self.connection:
                cursor = self.connection.cursor()
                cursor.execute(query, parameters)
                return cursor.rowcount
        except Exception as e:
            self.logger.error(f"Error al ejecutar actualización: {str(e)}")
            self.logger.error(f"Query: {query}, Params: {parameters}")
            raise

    def close(self) -> None:
        """Cierra la conexión SQLite del hilo actual."""
        self.close_connection()


class BaseModel:
    """Clase base para todos los modelos de datos."""
    def __init__(self):
        self.id = None
        
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> 'BaseModel':
        instance = cls()
        for key, value in row.items():
            setattr(instance, key, value)
        return instance

    def to_dict(self) -> Dict[str, Any]:
        # Filtrar atributos internos o métodos
        res = {}
        for key, value in self.__dict__.items():
            if key.startswith('_') or callable(value):
                continue
            res[key] = value
        # Añadir las propiedades calculadas que actúan como alias
        if hasattr(self, 'hip'):
            res['hip'] = self.hip
        if hasattr(self, 'shoulder_width'):
            res['shoulder_width'] = self.shoulder_width
        return res


class User(BaseModel):
    """Modelo que representa a un usuario del sistema."""
    def __init__(self, email: str = None, name: str = None, gender: str = "not_specified",
                 birth_date: str = None, height: float = None, weight: float = None,
                 username: str = None, first_name: str = None, last_name: str = None,
                 profile_image: str = None):
        super().__init__()
        self.email = email
        self.password_hash = None
        self.gender = gender
        self.birth_date = birth_date
        self.username = username or email
        self.first_name = first_name or (name.split(' ', 1)[0] if name else None)
        self.last_name = last_name or (name.split(' ', 1)[1] if name and ' ' in name else "")
        self.profile_image = profile_image
        self.created_at = datetime.datetime.now().isoformat()
        self.updated_at = datetime.datetime.now().isoformat()
        self.last_login = None

    @property
    def name(self) -> str:
        if self.first_name:
            return f"{self.first_name} {self.last_name}".strip()
        return self.username or ""

    @name.setter
    def name(self, value: str) -> None:
        if value:
            parts = value.split(' ', 1)
            self.first_name = parts[0]
            self.last_name = parts[1] if len(parts) > 1 else ""

    def set_password(self, password: str) -> None:
        """Establece la contraseña cifrada usando Werkzeug."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verifica la contraseña usando Werkzeug."""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    def update_last_login(self) -> None:
        self.last_login = datetime.datetime.now().isoformat()


class Measurement(BaseModel):
    """Modelo que representa las medidas corporales del usuario."""
    def __init__(self, user_id: int = None):
        super().__init__()
        self.user_id = user_id
        self.name = 'Mis medidas'
        self.height = None
        self.weight = None
        self.chest = None
        self.waist = None
        self.hips = None
        self.shoulders = None
        self.arm_length = None
        self.inseam = None
        self.neck = None
        self.thigh = None
        self.additional_measurements = "{}"
        self.front_image = None
        self.side_image = None
        self.landmarks = "{}"
        self.is_current = 1
        self.created_at = datetime.datetime.now().isoformat()
        self.updated_at = datetime.datetime.now().isoformat()

    # Propiedades para compatibilidad con código que usa los nombres antiguos (hip, shoulder_width, date, raw_data)
    @property
    def hip(self) -> Optional[float]:
        return self.hips

    @hip.setter
    def hip(self, value: Optional[float]) -> None:
        self.hips = value

    @property
    def shoulder_width(self) -> Optional[float]:
        return self.shoulders

    @shoulder_width.setter
    def shoulder_width(self, value: Optional[float]) -> None:
        self.shoulders = value

    @property
    def date(self) -> str:
        return self.created_at

    @date.setter
    def date(self, value: str) -> None:
        self.created_at = value

    @property
    def raw_data(self) -> str:
        # Crea un JSON con los landmarks y medidas para compatibilidad
        data = {
            "landmarks": json.loads(self.landmarks) if isinstance(self.landmarks, str) and self.landmarks else {},
            "measurements": {
                "height": self.height,
                "weight": self.weight,
                "chest": self.chest,
                "waist": self.waist,
                "hip": self.hips,
                "shoulder_width": self.shoulders,
                "arm_length": self.arm_length,
                "inseam": self.inseam,
                "neck": self.neck,
                "thigh": self.thigh
            }
        }
        return json.dumps(data)

    @raw_data.setter
    def raw_data(self, value: str) -> None:
        try:
            data = json.loads(value)
            if "landmarks" in data:
                self.landmarks = json.dumps(data["landmarks"])
            if "measurements" in data:
                m = data["measurements"]
                self.height = m.get("height", self.height)
                self.weight = m.get("weight", self.weight)
                self.chest = m.get("chest", self.chest)
                self.waist = m.get("waist", self.waist)
                self.hips = m.get("hip", self.hips)
                self.shoulders = m.get("shoulder_width", self.shoulders)
                self.arm_length = m.get("arm_length", self.arm_length)
                self.inseam = m.get("inseam", self.inseam)
                self.neck = m.get("neck", self.neck)
                self.thigh = m.get("thigh", self.thigh)
        except Exception:
            pass

    def from_landmarks(self, landmarks: Dict[str, Dict[str, float]], measurements: Dict[str, float]) -> None:
        """Actualiza las medidas a partir de landmarks y cálculos."""
        for key, value in measurements.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key == 'hip':
                self.hips = value
            elif key == 'shoulder_width':
                self.shoulders = value
        
        self.landmarks = json.dumps(landmarks)
        data = {
            "landmarks": landmarks,
            "measurements": measurements
        }
        self.additional_measurements = json.dumps(data)


class ClothingItem(BaseModel):
    """Modelo que representa una prenda de ropa."""
    def __init__(self, type: str = None, brand: str = None, name: str = None,
                 size: str = None, color: str = None, image_path: str = None):
        super().__init__()
        self.type = type
        self.brand = brand
        self.name = name
        self.size = size
        self.color = color
        self.image_path = image_path
        self.metadata = "{}"


class VirtualFitting(BaseModel):
    """Modelo que representa una prueba virtual de ropa."""
    def __init__(self, user_id: int = None, measurement_id: int = None):
        super().__init__()
        self.user_id = user_id
        self.measurement_id = measurement_id
        self.date = datetime.datetime.now().isoformat()
        self.result_image_path = None
        self.clothing_items = "[]"
        self.fit_scores = "{}"
        self.comments = None

    @property
    def clothing_id(self) -> Optional[int]:
        # Para compatibilidad con fitting_results
        try:
            items = json.loads(self.clothing_items)
            return items[0] if items else None
        except Exception:
            return None

    @clothing_id.setter
    def clothing_id(self, value: Optional[int]) -> None:
        if value is not None:
            self.clothing_items = json.dumps([value])

    @property
    def preview_image(self) -> Optional[str]:
        return self.result_image_path

    @preview_image.setter
    def preview_image(self, value: Optional[str]) -> None:
        self.result_image_path = value


class SizeRecommendation(BaseModel):
    """Modelo que representa una recomendación de talla."""
    def __init__(self, user_id: int = None, measurement_id: int = None,
                 clothing_type: str = None, brand: str = None):
        super().__init__()
        self.user_id = user_id
        self.measurement_id = measurement_id
        self.clothing_type = clothing_type
        self.brand = brand
        self.recommended_size = None
        self.fit_score = None
        self.details = "{}"
        self.date = datetime.datetime.now().isoformat()


# Wrapper y funciones globales para compatibilidad de app.py
def init_db():
    Database()

class SessionWrapper:
    def remove(self):
        Database().close_connection()

db_session = SessionWrapper()