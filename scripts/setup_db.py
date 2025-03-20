#!/usr/bin/env python3
"""
setup_db.py - Script para configurar la base de datos inicial del sistema de prueba virtual.

Este script:
- Crea las tablas necesarias para el sistema
- Inserta datos básicos iniciales
- Establece índices y relaciones
- Configura usuarios de prueba con medidas de ejemplo

Uso:
    python scripts/setup_db.py [--reset] [--samples]

Opciones:
    --reset    Elimina las tablas existentes antes de crear nuevas
    --samples  Carga datos de muestra para pruebas
"""

import os
import sys
import json
import argparse
import sqlite3
import hashlib
import datetime
from pathlib import Path

# Agregar directorio raíz al path para importar módulos del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Importar módulos del proyecto
from database.db import get_db_connection, DB_PATH

# Configuración
DEFAULT_DB_PATH = DB_PATH
TABLES_SCHEMA_PATH = ROOT_DIR / "database" / "schema.sql"
SAMPLE_DATA_PATH = ROOT_DIR / "data" / "references"


def parse_args():
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Configura la base de datos para Virtual Fitting Room")
    parser.add_argument("--reset", action="store_true", help="Elimina las tablas existentes antes de crear nuevas")
    parser.add_argument("--samples", action="store_true", help="Carga datos de ejemplo para pruebas")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Ruta a la base de datos SQLite")
    return parser.parse_args()


def create_schema(conn, reset=False):
    """Crea el esquema de la base de datos."""
    cursor = conn.cursor()
    
    # Opcionalmente eliminar tablas existentes
    if reset:
        print("Eliminando tablas existentes...")
        cursor.executescript("""
            PRAGMA foreign_keys = OFF;
            
            DROP TABLE IF EXISTS fitting_results;
            DROP TABLE IF EXISTS user_measurements;
            DROP TABLE IF EXISTS clothing_sizes;
            DROP TABLE IF EXISTS clothing_images;
            DROP TABLE IF EXISTS clothing;
            DROP TABLE IF EXISTS size_charts;
            DROP TABLE IF EXISTS brands;
            DROP TABLE IF EXISTS user_preferences;
            DROP TABLE IF EXISTS users;
            
            PRAGMA foreign_keys = ON;
        """)
    
    # Crear tablas
    print("Creando esquema de la base de datos...")
    
    # Tabla de usuarios
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        first_name TEXT,
        last_name TEXT,
        gender TEXT CHECK(gender IN ('male', 'female', 'other', 'not_specified')),
        birth_date DATE,
        profile_image TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    ''')
    
    # Tabla de preferencias de usuario
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        preferred_fit TEXT DEFAULT 'regular' CHECK(preferred_fit IN ('slim', 'regular', 'loose')),
        preferred_brands TEXT,
        preferred_categories TEXT,
        preferred_colors TEXT,
        size_region TEXT DEFAULT 'EU' CHECK(size_region IN ('EU', 'US', 'UK', 'INT')),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')
    
    # Tabla de marcas
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS brands (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        logo_url TEXT,
        website TEXT,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Tabla de tablas de tallas
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS size_charts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        brand_id INTEGER NOT NULL,
        gender TEXT NOT NULL CHECK(gender IN ('male', 'female', 'unisex')),
        category TEXT NOT NULL,
        chart_data TEXT NOT NULL, -- JSON con datos de la tabla de tallas
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (brand_id) REFERENCES brands(id) ON DELETE CASCADE,
        UNIQUE (brand_id, gender, category)
    )
    ''')
    
    # Tabla de prendas
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (clothing_id) REFERENCES clothing(id) ON DELETE CASCADE
    )
    ''')
    
    # Tabla de tallas de prendas
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS clothing_sizes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        clothing_id INTEGER NOT NULL,
        size_name TEXT NOT NULL,
        measurements TEXT NOT NULL, -- JSON con medidas específicas para esta talla
        stock INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (clothing_id) REFERENCES clothing(id) ON DELETE CASCADE,
        UNIQUE (clothing_id, size_name)
    )
    ''')
    
    # Tabla de medidas de usuarios
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT DEFAULT 'Mis medidas',
        height REAL, -- cm
        weight REAL, -- kg
        chest REAL, -- cm
        waist REAL, -- cm
        hips REAL, -- cm
        shoulders REAL, -- cm
        arm_length REAL, -- cm
        inseam REAL, -- cm
        neck REAL, -- cm
        additional_measurements TEXT, -- JSON con medidas adicionales
        front_image TEXT, -- ruta a la imagen frontal
        side_image TEXT, -- ruta a la imagen lateral
        landmarks TEXT, -- JSON con landmarks detectados
        is_current BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')
    
    # Tabla de resultados de ajuste
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fitting_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        clothing_id INTEGER NOT NULL,
        size_name TEXT NOT NULL,
        measurement_id INTEGER NOT NULL,
        fit_score REAL NOT NULL, -- puntuación de 0 a 100
        fit_type TEXT CHECK(fit_type IN ('tight', 'regular', 'loose')),
        fit_details TEXT, -- JSON con detalles de ajuste por área
        preview_image TEXT, -- ruta a la imagen de visualización
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (clothing_id) REFERENCES clothing(id) ON DELETE CASCADE,
        FOREIGN KEY (measurement_id) REFERENCES user_measurements(id) ON DELETE CASCADE
    )
    ''')
    
    # Crear índices para optimizar búsquedas
    print("Creando índices...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_measurements_user_id ON user_measurements(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clothing_brand_id ON clothing(brand_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clothing_category ON clothing(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clothing_sizes_clothing_id ON clothing_sizes(clothing_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fitting_results_user_id ON fitting_results(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fitting_results_clothing_id ON fitting_results(clothing_id)")
    
    # Crear disparadores para actualizar timestamps
    print("Configurando disparadores para timestamps...")
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS update_user_timestamp 
    AFTER UPDATE ON users
    FOR EACH ROW
    BEGIN
        UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
    END;
    ''')
    
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS update_clothing_timestamp 
    AFTER UPDATE ON clothing
    FOR EACH ROW
    BEGIN
        UPDATE clothing SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
    END;
    ''')
    
    # Confirmar cambios
    conn.commit()
    print("Esquema de base de datos creado correctamente.")


def load_sample_data(conn):
    """Carga datos de muestra en la base de datos."""
    cursor = conn.cursor()
    
    print("Cargando datos de muestra...")
    
    # Crear usuarios de prueba
    print("Creando usuarios de prueba...")
    users = [
        {
            "username": "usuario_prueba",
            "email": "usuario@ejemplo.com",
            "password": "contraseña123",
            "first_name": "Usuario",
            "last_name": "De Prueba",
            "gender": "male"
        },
        {
            "username": "maria_garcia",
            "email": "maria@ejemplo.com",
            "password": "maria123",
            "first_name": "María",
            "last_name": "García",
            "gender": "female"
        }
    ]
    
    for user in users:
        # Hash de contraseña simple (en producción usar bcrypt o similar)
        password_hash = hashlib.sha256(user["password"].encode()).hexdigest()
        
        cursor.execute(
            '''
            INSERT OR IGNORE INTO users 
            (username, email, password_hash, first_name, last_name, gender)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (user["username"], user["email"], password_hash, 
             user["first_name"], user["last_name"], user["gender"])
        )
    
    # Insertar marcas
    print("Cargando marcas...")
    brands = [
        {"name": "Zara", "logo_url": "/static/img/brands/zara.png", "website": "https://www.zara.com"},
        {"name": "H&M", "logo_url": "/static/img/brands/hm.png", "website": "https://www.hm.com"},
        {"name": "Levi's", "logo_url": "/static/img/brands/levis.png", "website": "https://www.levi.com"},
        {"name": "Nike", "logo_url": "/static/img/brands/nike.png", "website": "https://www.nike.com"},
        {"name": "Adidas", "logo_url": "/static/img/brands/adidas.png", "website": "https://www.adidas.com"},
        {"name": "Mango", "logo_url": "/static/img/brands/mango.png", "website": "https://www.mango.com"},
        {"name": "Generic", "logo_url": "/static/img/brands/generic.png", "website": ""}
    ]
    
    for brand in brands:
        cursor.execute(
            "INSERT OR IGNORE INTO brands (name, logo_url, website) VALUES (?, ?, ?)",
            (brand["name"], brand["logo_url"], brand["website"])
        )
    
    # Cargar tablas de tallas desde archivos JSON
    print("Cargando tablas de tallas...")
    # Obtener IDs de las marcas
    cursor.execute("SELECT id, name FROM brands")
    brand_ids = {row[1]: row[0] for row in cursor.fetchall()}
    
    # Cargar tabla genérica
    for gender in ["male", "female"]:
        try:
            with open(SAMPLE_DATA_PATH / f"size_charts/generic_{gender}.json", "r") as f:
                chart_data = json.load(f)
                
                for category, data in chart_data.items():
                    cursor.execute(
                        '''
                        INSERT OR REPLACE INTO size_charts 
                        (brand_id, gender, category, chart_data)
                        VALUES (?, ?, ?, ?)
                        ''',
                        (brand_ids["Generic"], gender, category, json.dumps(data))
                    )
        except FileNotFoundError:
            print(f"Advertencia: No se encontró la tabla de tallas genérica para {gender}")
    
    # Cargar tablas específicas de marcas
    for brand_name in ["zara", "hm", "levis"]:
        if brand_name.title() not in brand_ids:
            continue
        
        try:
            with open(SAMPLE_DATA_PATH / f"size_charts/{brand_name}.json", "r") as f:
                brand_data = json.load(f)
                
                for gender, categories in brand_data.items():
                    for category, data in categories.items():
                        cursor.execute(
                            '''
                            INSERT OR REPLACE INTO size_charts 
                            (brand_id, gender, category, chart_data)
                            VALUES (?, ?, ?, ?)
                            ''',
                            (brand_ids[brand_name.title()], gender, category, json.dumps(data))
                        )
        except FileNotFoundError:
            print(f"Advertencia: No se encontró la tabla de tallas para {brand_name}")
    
    # Insertar prendas de muestra
    print("Cargando prendas de muestra...")
    clothing_items = [
        {
            "name": "Camisa Oxford Azul",
            "brand": "Zara",
            "category": "tops",
            "subcategory": "shirts",
            "type": "shirt",
            "gender": "male",
            "description": "Camisa Oxford de algodón con corte regular y botones en el cuello.",
            "color": "blue",
            "material": "100% algodón",
            "price": 29.99,
            "images": [
                {"url": "/data/clothes/shirts/shirt_blue_M.png", "type": "main"},
                {"url": "/data/clothes/shirts/shirt_blue_M_back.png", "type": "back"}
            ],
            "sizes": [
                {"name": "S", "measurements": {"chest": 100, "shoulder": 44, "sleeve": 64, "length": 70}},
                {"name": "M", "measurements": {"chest": 106, "shoulder": 46, "sleeve": 65, "length": 72}},
                {"name": "L", "measurements": {"chest": 112, "shoulder": 48, "sleeve": 66, "length": 74}},
                {"name": "XL", "measurements": {"chest": 118, "shoulder": 50, "sleeve": 67, "length": 76}}
            ]
        },
        {
            "name": "Camiseta Básica Blanca",
            "brand": "H&M",
            "category": "tops",
            "subcategory": "tshirts",
            "type": "t-shirt",
            "gender": "unisex",
            "description": "Camiseta básica de algodón con cuello redondo.",
            "color": "white",
            "material": "100% algodón",
            "price": 12.99,
            "images": [
                {"url": "/data/clothes/shirts/shirt_white_L.png", "type": "main"}
            ],
            "sizes": [
                {"name": "S", "measurements": {"chest": 96, "shoulder": 43, "length": 68}},
                {"name": "M", "measurements": {"chest": 102, "shoulder": 45, "length": 70}},
                {"name": "L", "measurements": {"chest": 108, "shoulder": 47, "length": 72}},
                {"name": "XL", "measurements": {"chest": 114, "shoulder": 49, "length": 74}}
            ]
        },
        {
            "name": "Pantalón Chino Beige",
            "brand": "Levi's",
            "category": "bottoms",
            "subcategory": "pants",
            "type": "pants",
            "gender": "male",
            "description": "Pantalón chino de corte slim en color beige.",
            "color": "beige",
            "material": "98% algodón, 2% elastano",
            "price": 49.99,
            "images": [
                {"url": "/data/clothes/pants/pants_beige_30.png", "type": "main"}
            ],
            "sizes": [
                {"name": "28", "measurements": {"waist": 72, "hips": 90, "inseam": 78}},
                {"name": "30", "measurements": {"waist": 76, "hips": 94, "inseam": 79}},
                {"name": "32", "measurements": {"waist": 81, "hips": 98, "inseam": 80}},
                {"name": "34", "measurements": {"waist": 86, "hips": 102, "inseam": 81}}
            ]
        },
        {
            "name": "Vaquero Azul Oscuro",
            "brand": "Levi's",
            "category": "bottoms",
            "subcategory": "jeans",
            "type": "pants",
            "gender": "male",
            "description": "Vaquero clásico 501 en azul oscuro con corte regular.",
            "color": "dark blue",
            "material": "100% algodón",
            "price": 89.99,
            "images": [
                {"url": "/data/clothes/pants/pants_blue_34.png", "type": "main"}
            ],
            "sizes": [
                {"name": "30", "measurements": {"waist": 76, "hips": 94, "inseam": 79}},
                {"name": "32", "measurements": {"waist": 81, "hips": 98, "inseam": 80}},
                {"name": "34", "measurements": {"waist": 86, "hips": 102, "inseam": 81}},
                {"name": "36", "measurements": {"waist": 91, "hips": 106, "inseam": 82}}
            ]
        },
        {
            "name": "Vestido Midi Floral",
            "brand": "Zara",
            "category": "dresses",
            "subcategory": "midi",
            "type": "dress",
            "gender": "female",
            "description": "Vestido midi con estampado floral y manga corta.",
            "color": "floral",
            "material": "100% viscosa",
            "price": 39.99,
            "images": [
                {"url": "/data/clothes/dresses/dress_floral_M.png", "type": "main"}
            ],
            "sizes": [
                {"name": "XS", "measurements": {"chest": 84, "waist": 66, "hips": 91, "length": 110}},
                {"name": "S", "measurements": {"chest": 88, "waist": 70, "hips": 95, "length": 112}},
                {"name": "M", "measurements": {"chest": 94, "waist": 76, "hips": 101, "length": 114}},
                {"name": "L", "measurements": {"chest": 100, "waist": 83, "hips": 107, "length": 116}}
            ]
        }
    ]
    
    for item in clothing_items:
        # Insertar prenda
        cursor.execute(
            '''
            INSERT INTO clothing 
            (name, brand_id, category, subcategory, type, gender, description, color, material, price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                item["name"], 
                brand_ids.get(item["brand"], None),
                item["category"],
                item["subcategory"],
                item["type"],
                item["gender"],
                item["description"],
                item["color"],
                item["material"],
                item["price"]
            )
        )
        
        clothing_id = cursor.lastrowid
        
        # Insertar imágenes
        for image in item["images"]:
            cursor.execute(
                "INSERT INTO clothing_images (clothing_id, image_url, image_type) VALUES (?, ?, ?)",
                (clothing_id, image["url"], image["type"])
            )
        
        # Insertar tallas
        for size in item["sizes"]:
            cursor.execute(
                "INSERT INTO clothing_sizes (clothing_id, size_name, measurements) VALUES (?, ?, ?)",
                (clothing_id, size["name"], json.dumps(size["measurements"]))
            )
    
    # Insertar medidas de usuario de ejemplo
    print("Creando medidas de usuario de muestra...")
    # Obtener IDs de usuarios
    cursor.execute("SELECT id, username FROM users")
    user_ids = {row[1]: row[0] for row in cursor.fetchall()}
    
    # Medidas para usuario masculino
    if "usuario_prueba" in user_ids:
        cursor.execute(
            '''
            INSERT INTO user_measurements 
            (user_id, name, height, weight, chest, waist, hips, shoulders, arm_length, inseam, neck)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                user_ids["usuario_prueba"],
                "Mis medidas principales",
                178.0,  # altura en cm
                75.0,   # peso en kg
                102.0,  # pecho
                88.0,   # cintura
                98.0,   # cadera
                46.0,   # hombros
                67.0,   # longitud de brazo
                82.0,   # entrepierna
                40.0    # cuello
            )
        )
    
    # Medidas para usuaria femenina
    if "maria_garcia" in user_ids:
        cursor.execute(
            '''
            INSERT INTO user_measurements 
            (user_id, name, height, weight, chest, waist, hips, shoulders, arm_length, inseam, neck)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                user_ids["maria_garcia"],
                "Mis medidas actuales",
                165.0,  # altura en cm
                62.0,   # peso en kg
                92.0,   # pecho
                74.0,   # cintura
                98.0,   # cadera
                42.0,   # hombros
                62.0,   # longitud de brazo
                78.0,   # entrepierna
                34.0    # cuello
            )
        )
    
    # Insertar preferencias de usuario
    print("Configurando preferencias de usuario...")
    if "usuario_prueba" in user_ids:
        cursor.execute(
            '''
            INSERT OR REPLACE INTO user_preferences
            (user_id, preferred_fit, preferred_brands, preferred_categories, size_region)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (
                user_ids["usuario_prueba"],
                "regular",
                json.dumps(["Levi's", "Nike"]),
                json.dumps(["shirts", "pants"]),
                "EU"
            )
        )
    
    if "maria_garcia" in user_ids:
        cursor.execute(
            '''
            INSERT OR REPLACE INTO user_preferences
            (user_id, preferred_fit, preferred_brands, preferred_categories, size_region)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (
                user_ids["maria_garcia"],
                "slim",
                json.dumps(["Zara", "Mango"]),
                json.dumps(["dresses", "tops"]),
                "EU"
            )
        )
    
    # Insertar algunos resultados de ajuste de ejemplo
    print("Creando resultados de ajuste de ejemplo...")
    # Obtener IDs de prendas, tallas y medidas
    cursor.execute("SELECT id, name FROM clothing")
    clothing_ids = {row[1]: row[0] for row in cursor.fetchall()}
    
    cursor.execute("SELECT id, user_id FROM user_measurements")
    measurement_ids = {}
    for row in cursor.fetchall():
        user_id = row[1]
        if user_id not in measurement_ids:
            measurement_ids[user_id] = row[0]
    
    # Resultados para usuario masculino
    if "usuario_prueba" in user_ids and "Camisa Oxford Azul" in clothing_ids:
        user_id = user_ids["usuario_prueba"]
        clothing_id = clothing_ids["Camisa Oxford Azul"]
        measurement_id = measurement_ids.get(user_id)
        
        if measurement_id:
            cursor.execute(
                '''
                INSERT INTO fitting_results
                (user_id, clothing_id, size_name, measurement_id, fit_score, fit_type, fit_details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    user_id,
                    clothing_id,
                    "L",  # talla recomendada
                    measurement_id,
                    85.5,  # puntuación de ajuste
                    "regular",
                    json.dumps({
                        "chest": {"fit": "regular", "score": 90},
                        "shoulder": {"fit": "regular", "score": 88},
                        "sleeve": {"fit": "tight", "score": 78},
                        "length": {"fit": "regular", "score": 86}
                    })
                )
            )
    
    # Resultados para usuaria femenina
    if "maria_garcia" in user_ids and "Vestido Midi Floral" in clothing_ids:
        user_id = user_ids["maria_garcia"]
        clothing_id = clothing_ids["Vestido Midi Floral"]
        measurement_id = measurement_ids.get(user_id)
        
        if measurement_id:
            cursor.execute(
                '''
                INSERT INTO fitting_results
                (user_id, clothing_id, size_name, measurement_id, fit_score, fit_type, fit_details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    user_id,
                    clothing_id,
                    "M",  # talla recomendada
                    measurement_id,
                    92.0,  # puntuación de ajuste
                    "regular",
                    json.dumps({
                        "chest": {"fit": "regular", "score": 94},
                        "waist": {"fit": "regular", "score": 92},
                        "hips": {"fit": "regular", "score": 90},
                        "length": {"fit": "regular", "score": 92}
                    })
                )
            )
    
    # Confirmar cambios
    conn.commit()
    print("Datos de muestra cargados correctamente.")


def main():
    """Función principal del script."""
    args = parse_args()
    
    # Crear directorio para la base de datos si no existe
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    
    print(f"Configurando base de datos en: {args.db_path}")
    
    # Inicializar conexión
    conn = sqlite3.connect(args.db_path)
    
    try:
        # Crear esquema
        create_schema(conn, reset=args.reset)
        
        # Cargar datos de muestra si se solicita
        if args.samples:
            load_sample_data(conn)
        
        print("Configuración de base de datos completada exitosamente.")
    except Exception as e:
        print(f"Error durante la configuración de la base de datos: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()