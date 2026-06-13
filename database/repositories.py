"""
Módulo de repositorios para el acceso a datos del sistema de prueba virtual.
Implementa el patrón repositorio para aislar la lógica de negocio del acceso a datos.
"""

import json
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime

from .db import (
    Database, 
    User, 
    Measurement, 
    ClothingItem, 
    VirtualFitting,
    SizeRecommendation
)


class BaseRepository:
    """
    Clase base para todos los repositorios.
    Proporciona funcionalidad común.
    """
    
    def __init__(self):
        self.db = Database()
        self.logger = logging.getLogger(__name__)
        
    def _execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        try:
            return self.db.execute_query(query, params)
        except Exception as e:
            self.logger.error(f"Error en consulta: {e}")
            return []
            
    def _execute_insert(self, query: str, params: tuple = ()) -> int:
        try:
            return self.db.execute_insert(query, params)
        except Exception as e:
            self.logger.error(f"Error en inserción: {e}")
            return -1
            
    def _execute_update(self, query: str, params: tuple = ()) -> int:
        try:
            return self.db.execute_update(query, params)
        except Exception as e:
            self.logger.error(f"Error en actualización: {e}")
            return 0


class UserRepository(BaseRepository):
    """Repositorio para operaciones relacionadas con usuarios."""
    
    def create(self, user: User) -> int:
        query = '''
        INSERT INTO users (username, email, password_hash, first_name, last_name, gender, birth_date, 
                           profile_image, created_at, updated_at, last_login)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (user.username, user.email, user.password_hash, user.first_name, user.last_name, 
                  user.gender, user.birth_date, user.profile_image, user.created_at, user.updated_at, 
                  user.last_login)
                  
        user_id = self._execute_insert(query, params)
        if user_id > 0:
            user.id = user_id
        return user_id
    
    def get_by_id(self, user_id: int) -> Optional[User]:
        query = "SELECT * FROM users WHERE id = ?"
        params = (user_id,)
        
        results = self._execute_query(query, params)
        if results:
            return User.from_row(results[0])
        return None
    
    def get_by_email(self, email: str) -> Optional[User]:
        query = "SELECT * FROM users WHERE email = ?"
        params = (email,)
        
        results = self._execute_query(query, params)
        if results:
            return User.from_row(results[0])
        return None
    
    def update(self, user: User) -> bool:
        user.updated_at = datetime.now().isoformat()
        query = '''
        UPDATE users 
        SET username = ?, email = ?, first_name = ?, last_name = ?, gender = ?, birth_date = ?, 
            profile_image = ?, updated_at = ?, last_login = ?
        WHERE id = ?
        '''
        params = (user.username, user.email, user.first_name, user.last_name, user.gender, user.birth_date,
                  user.profile_image, user.updated_at, user.last_login, user.id)
                  
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def update_password(self, user_id: int, password_hash: str) -> bool:
        now = datetime.now().isoformat()
        query = "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?"
        params = (password_hash, now, user_id)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, user_id: int) -> bool:
        query = "DELETE FROM users WHERE id = ?"
        params = (user_id,)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[User]:
        query = "SELECT * FROM users ORDER BY username LIMIT ? OFFSET ?"
        params = (limit, offset)
        
        results = self._execute_query(query, params)
        return [User.from_row(row) for row in results]
    
    def update_last_login(self, user_id: int) -> bool:
        now = datetime.now().isoformat()
        query = "UPDATE users SET last_login = ? WHERE id = ?"
        params = (now, user_id)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0


class MeasurementRepository(BaseRepository):
    """Repositorio para operaciones relacionadas con medidas corporales (user_measurements)."""
    
    def create(self, measurement: Measurement) -> int:
        query = '''
        INSERT INTO user_measurements (user_id, name, height, weight, chest, waist, hips,
                                     shoulders, arm_length, inseam, neck, thigh, additional_measurements,
                                     front_image, side_image, landmarks, is_current, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (measurement.user_id, measurement.name, measurement.height,
                  measurement.weight, measurement.chest, measurement.waist,
                  measurement.hips, measurement.shoulders, measurement.arm_length,
                  measurement.inseam, measurement.neck, measurement.thigh,
                  measurement.additional_measurements, measurement.front_image,
                  measurement.side_image, measurement.landmarks, measurement.is_current,
                  measurement.created_at, measurement.updated_at)
                  
        measurement_id = self._execute_insert(query, params)
        if measurement_id > 0:
            measurement.id = measurement_id
        return measurement_id
    
    def get_by_id(self, measurement_id: int) -> Optional[Measurement]:
        query = "SELECT * FROM user_measurements WHERE id = ?"
        params = (measurement_id,)
        
        results = self._execute_query(query, params)
        if results:
            return Measurement.from_row(results[0])
        return None
    
    def get_latest_for_user(self, user_id: int) -> Optional[Measurement]:
        query = "SELECT * FROM user_measurements WHERE user_id = ? ORDER BY created_at DESC LIMIT 1"
        params = (user_id,)
        
        results = self._execute_query(query, params)
        if results:
            return Measurement.from_row(results[0])
        return None
    
    def get_all_for_user(self, user_id: int, limit: int = 10) -> List[Measurement]:
        query = "SELECT * FROM user_measurements WHERE user_id = ? ORDER BY created_at DESC LIMIT ?"
        params = (user_id, limit)
        
        results = self._execute_query(query, params)
        return [Measurement.from_row(row) for row in results]
    
    def update(self, measurement: Measurement) -> bool:
        measurement.updated_at = datetime.now().isoformat()
        query = '''
        UPDATE user_measurements 
        SET name = ?, height = ?, weight = ?, chest = ?, waist = ?, hips = ?,
            shoulders = ?, arm_length = ?, inseam = ?, neck = ?, thigh = ?, 
            additional_measurements = ?, front_image = ?, side_image = ?, 
            landmarks = ?, is_current = ?, updated_at = ?
        WHERE id = ?
        '''
        params = (measurement.name, measurement.height, measurement.weight, measurement.chest,
                  measurement.waist, measurement.hips, measurement.shoulders, measurement.arm_length,
                  measurement.inseam, measurement.neck, measurement.thigh, measurement.additional_measurements,
                  measurement.front_image, measurement.side_image, measurement.landmarks,
                  measurement.is_current, measurement.updated_at, measurement.id)
                  
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, measurement_id: int) -> bool:
        query = "DELETE FROM user_measurements WHERE id = ?"
        params = (measurement_id,)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def get_history(self, user_id: int, metric: str) -> List[Dict[str, Any]]:
        # Mapear nombres antiguos a columnas de user_measurements
        db_metric = metric
        if metric == "hip":
            db_metric = "hips"
        elif metric == "shoulder_width":
            db_metric = "shoulders"
            
        if db_metric not in ["height", "weight", "chest", "waist", "hips", 
                             "shoulders", "arm_length", "inseam", "neck", "thigh"]:
            self.logger.warning(f"Métrica inválida: {metric}")
            return []
            
        query = f"SELECT created_at as date, {db_metric} FROM user_measurements WHERE user_id = ? ORDER BY created_at"
        params = (user_id,)
        
        results = self._execute_query(query, params)
        return [{"date": row["date"], "value": row[db_metric]} for row in results if row[db_metric] is not None]


class ClothingRepository(BaseRepository):
    """Repositorio para operaciones relacionadas con prendas de ropa (clothing)."""
    
    def create(self, clothing: ClothingItem) -> int:
        # Nota: Este repositorio ahora mapea a la tabla clothing estructurada
        # Primero buscar o crear la marca correspondiente
        brand_query = "SELECT id FROM brands WHERE name = ?"
        brand_results = self._execute_query(brand_query, (clothing.brand,))
        if brand_results:
            brand_id = brand_results[0]["id"]
        else:
            brand_id = self._execute_insert("INSERT INTO brands (name) VALUES (?)", (clothing.brand,))

        # Insertar en tabla clothing
        query = '''
        INSERT INTO clothing (type, brand_id, name, color, description, price, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        '''
        params = (clothing.type, brand_id, clothing.name, clothing.color, 
                  clothing.metadata, 0.0) # metadata se usa como fallback para description
                  
        clothing_id = self._execute_insert(query, params)
        if clothing_id > 0:
            clothing.id = clothing_id
            # Guardar la imagen principal
            if clothing.image_path:
                self._execute_insert(
                    "INSERT INTO clothing_images (clothing_id, image_url, image_type) VALUES (?, ?, 'main')",
                    (clothing_id, clothing.image_path)
                )
            # Guardar talla
            if clothing.size:
                self._execute_insert(
                    "INSERT INTO clothing_sizes (clothing_id, size_name, measurements) VALUES (?, ?, ?)",
                    (clothing_id, clothing.size, "{}")
                )
        return clothing_id
    
    def get_by_id(self, clothing_id: int) -> Optional[ClothingItem]:
        query = '''
        SELECT c.id, c.type, b.name as brand, c.name, c.color, ci.image_url as image_path, 
               c.description, c.price
        FROM clothing c
        LEFT JOIN brands b ON c.brand_id = b.id
        LEFT JOIN clothing_images ci ON c.id = ci.clothing_id AND ci.image_type = 'main'
        WHERE c.id = ?
        '''
        params = (clothing_id,)
        
        results = self._execute_query(query, params)
        if results:
            row = results[0]
            # Obtener tallas de la prenda
            sizes_results = self._execute_query("SELECT size_name FROM clothing_sizes WHERE clothing_id = ?", (clothing_id,))
            size_str = ",".join([r["size_name"] for r in sizes_results]) if sizes_results else "M"
            
            item = ClothingItem()
            item.id = row["id"]
            item.type = row["type"]
            item.brand = row["brand"] or "Generic"
            item.name = row["name"]
            item.size = size_str
            item.color = row["color"]
            item.image_path = row["image_path"] or ""
            # Guardar descripción y precio en metadatos para compatibilidad
            meta = {
                "description": row["description"] or "",
                "price": row["price"] or 0.0
            }
            item.metadata = json.dumps(meta)
            return item
        return None

    def get_all_for_api(self) -> List[Dict[str, Any]]:
        """Devuelve todas las prendas con detalles para la API del frontend."""
        query = '''
        SELECT c.id, c.name, b.name as brand, ci.image_url as thumbnail, c.price, c.description, c.type
        FROM clothing c
        LEFT JOIN brands b ON c.brand_id = b.id
        LEFT JOIN clothing_images ci ON c.id = ci.clothing_id AND ci.image_type = 'main'
        '''
        results = self._execute_query(query)
        # Asegurar fallbacks si no hay imágenes o marcas
        for row in results:
            if not row["thumbnail"]:
                # Generar ruta por defecto según tipo
                row["thumbnail"] = f"/static/img/clothes/{row['type']}_default.png"
            if not row["brand"]:
                row["brand"] = "Generic"
            if not row["price"]:
                row["price"] = 29.99
            if not row["description"]:
                row["description"] = "Prenda de excelente calidad para probador virtual."
        return results
    
    def get_by_type(self, clothing_type: str, limit: int = 50, offset: int = 0) -> List[ClothingItem]:
        query = '''
        SELECT c.id, c.type, b.name as brand, c.name, c.color, ci.image_url as image_path, c.description, c.price
        FROM clothing c
        LEFT JOIN brands b ON c.brand_id = b.id
        LEFT JOIN clothing_images ci ON c.id = ci.clothing_id AND ci.image_type = 'main'
        WHERE c.type = ? LIMIT ? OFFSET ?
        '''
        params = (clothing_type, limit, offset)
        
        results = self._execute_query(query, params)
        items = []
        for row in results:
            item = ClothingItem()
            item.id = row["id"]
            item.type = row["type"]
            item.brand = row["brand"] or "Generic"
            item.name = row["name"]
            item.color = row["color"]
            item.image_path = row["image_path"] or ""
            meta = {"description": row["description"] or "", "price": row["price"] or 0.0}
            item.metadata = json.dumps(meta)
            items.append(item)
        return items
    
    def get_by_brand(self, brand: str, limit: int = 50, offset: int = 0) -> List[ClothingItem]:
        query = '''
        SELECT c.id, c.type, b.name as brand, c.name, c.color, ci.image_url as image_path, c.description, c.price
        FROM clothing c
        LEFT JOIN brands b ON c.brand_id = b.id
        LEFT JOIN clothing_images ci ON c.id = ci.clothing_id AND ci.image_type = 'main'
        WHERE b.name = ? LIMIT ? OFFSET ?
        '''
        params = (brand, limit, offset)
        
        results = self._execute_query(query, params)
        items = []
        for row in results:
            item = ClothingItem()
            item.id = row["id"]
            item.type = row["type"]
            item.brand = row["brand"] or "Generic"
            item.name = row["name"]
            item.color = row["color"]
            item.image_path = row["image_path"] or ""
            meta = {"description": row["description"] or "", "price": row["price"] or 0.0}
            item.metadata = json.dumps(meta)
            items.append(item)
        return items
    
    def get_by_type_and_brand(self, clothing_type: str, brand: str) -> List[ClothingItem]:
        query = '''
        SELECT c.id, c.type, b.name as brand, c.name, c.color, ci.image_url as image_path, c.description, c.price
        FROM clothing c
        LEFT JOIN brands b ON c.brand_id = b.id
        LEFT JOIN clothing_images ci ON c.id = ci.clothing_id AND ci.image_type = 'main'
        WHERE c.type = ? AND b.name = ?
        '''
        params = (clothing_type, brand)
        
        results = self._execute_query(query, params)
        items = []
        for row in results:
            item = ClothingItem()
            item.id = row["id"]
            item.type = row["type"]
            item.brand = row["brand"] or "Generic"
            item.name = row["name"]
            item.color = row["color"]
            item.image_path = row["image_path"] or ""
            meta = {"description": row["description"] or "", "price": row["price"] or 0.0}
            item.metadata = json.dumps(meta)
            items.append(item)
        return items
    
    def update(self, clothing: ClothingItem) -> bool:
        # Buscar el brand_id
        brand_query = "SELECT id FROM brands WHERE name = ?"
        brand_results = self._execute_query(brand_query, (clothing.brand,))
        if brand_results:
            brand_id = brand_results[0]["id"]
        else:
            brand_id = self._execute_insert("INSERT INTO brands (name) VALUES (?)", (clothing.brand,))

        # Update clothing
        query = '''
        UPDATE clothing 
        SET type = ?, brand_id = ?, name = ?, color = ?, description = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        '''
        params = (clothing.type, brand_id, clothing.name, clothing.color, clothing.metadata, clothing.id)
        rows_affected = self._execute_update(query, params)
        
        # Update image
        if clothing.image_path:
            self._execute_update(
                "INSERT OR REPLACE INTO clothing_images (clothing_id, image_url, image_type) VALUES (?, ?, 'main')",
                (clothing.id, clothing.image_path)
            )
        return rows_affected > 0
    
    def delete(self, clothing_id: int) -> bool:
        query = "DELETE FROM clothing WHERE id = ?"
        params = (clothing_id,)
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def search(self, search_term: str, limit: int = 20) -> List[ClothingItem]:
        query = '''
        SELECT c.id, c.type, b.name as brand, c.name, c.color, ci.image_url as image_path, c.description, c.price
        FROM clothing c
        LEFT JOIN brands b ON c.brand_id = b.id
        LEFT JOIN clothing_images ci ON c.id = ci.clothing_id AND ci.image_type = 'main'
        WHERE c.name LIKE ? OR b.name LIKE ? OR c.type LIKE ? OR c.color LIKE ?
        LIMIT ?
        '''
        pattern = f"%{search_term}%"
        params = (pattern, pattern, pattern, pattern, limit)
        
        results = self._execute_query(query, params)
        items = []
        for row in results:
            item = ClothingItem()
            item.id = row["id"]
            item.type = row["type"]
            item.brand = row["brand"] or "Generic"
            item.name = row["name"]
            item.color = row["color"]
            item.image_path = row["image_path"] or ""
            meta = {"description": row["description"] or "", "price": row["price"] or 0.0}
            item.metadata = json.dumps(meta)
            items.append(item)
        return items
    
    def get_available_brands(self) -> List[str]:
        query = "SELECT DISTINCT name FROM brands ORDER BY name"
        results = self._execute_query(query)
        return [row["name"] for row in results if row["name"]]
    
    def get_available_types(self) -> List[str]:
        query = "SELECT DISTINCT type FROM clothing ORDER BY type"
        results = self._execute_query(query)
        return [row["type"] for row in results if row["type"]]
    
    def get_available_colors(self) -> List[str]:
        query = "SELECT DISTINCT color FROM clothing ORDER BY color"
        results = self._execute_query(query)
        return [row["color"] for row in results if row["color"]]


class FittingRepository(BaseRepository):
    """Repositorio para operaciones relacionadas con pruebas virtuales (fitting_results)."""
    
    def create(self, fitting: VirtualFitting) -> int:
        query = '''
        INSERT INTO fitting_results (user_id, clothing_id, size_name, measurement_id, fit_score, fit_type, fit_details, preview_image)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        # Obtener los detalles de ajuste e ID de prenda
        clothing_id = fitting.clothing_id or 1
        size_name = "M"
        try:
            scores = json.loads(fitting.fit_scores)
            fit_score = scores.get("general", 85.0)
        except Exception:
            fit_score = 85.0
            
        params = (fitting.user_id, clothing_id, size_name, fitting.measurement_id,
                  fit_score, "regular", fitting.fit_scores, fitting.result_image_path)
                  
        fitting_id = self._execute_insert(query, params)
        if fitting_id > 0:
            fitting.id = fitting_id
        return fitting_id
    
    def get_by_id(self, fitting_id: int) -> Optional[VirtualFitting]:
        query = "SELECT * FROM fitting_results WHERE id = ?"
        params = (fitting_id,)
        
        results = self._execute_query(query, params)
        if results:
            row = results[0]
            fitting = VirtualFitting()
            fitting.id = row["id"]
            fitting.user_id = row["user_id"]
            fitting.measurement_id = row["measurement_id"]
            fitting.date = row["created_at"]
            fitting.result_image_path = row["preview_image"]
            fitting.clothing_id = row["clothing_id"]
            fitting.fit_scores = row["fit_details"]
            fitting.comments = f"Ajuste {row['fit_type']} con score {row['fit_score']}"
            return fitting
        return None
    
    def get_by_user(self, user_id: int, limit: int = 10) -> List[VirtualFitting]:
        query = "SELECT * FROM fitting_results WHERE user_id = ? ORDER BY created_at DESC LIMIT ?"
        params = (user_id, limit)
        
        results = self._execute_query(query, params)
        fittings = []
        for row in results:
            fitting = VirtualFitting()
            fitting.id = row["id"]
            fitting.user_id = row["user_id"]
            fitting.measurement_id = row["measurement_id"]
            fitting.date = row["created_at"]
            fitting.result_image_path = row["preview_image"]
            fitting.clothing_id = row["clothing_id"]
            fitting.fit_scores = row["fit_details"]
            fittings.append(fitting)
        return fittings
    
    def update(self, fitting: VirtualFitting) -> bool:
        # Obtener los detalles de ajuste
        size_name = "M"
        try:
            scores = json.loads(fitting.fit_scores)
            fit_score = scores.get("general", 85.0)
        except Exception:
            fit_score = 85.0

        query = '''
        UPDATE fitting_results 
        SET preview_image = ?, fit_details = ?, fit_score = ?
        WHERE id = ?
        '''
        params = (fitting.result_image_path, fitting.fit_scores, fit_score, fitting.id)
                  
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, fitting_id: int) -> bool:
        query = "DELETE FROM fitting_results WHERE id = ?"
        params = (fitting_id,)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def get_user_history(self, user_id: int, full_details: bool = False) -> List[Dict[str, Any]]:
        if not full_details:
            query = "SELECT * FROM fitting_results WHERE user_id = ? ORDER BY created_at DESC"
            params = (user_id,)
            return self._execute_query(query, params)
        
        query = '''
        SELECT fr.id, fr.created_at as date, fr.preview_image as result_image_path,
               c.name as clothing_name, b.name as clothing_brand, fr.fit_score, fr.fit_type
        FROM fitting_results fr
        LEFT JOIN clothing c ON fr.clothing_id = c.id
        LEFT JOIN brands b ON c.brand_id = b.id
        WHERE fr.user_id = ?
        ORDER BY fr.created_at DESC
        '''
        params = (user_id,)
        return self._execute_query(query, params)


class RecommendationRepository(BaseRepository):
    """Repositorio para operaciones relacionadas con recomendaciones de tallas."""
    
    def create(self, recommendation: SizeRecommendation) -> int:
        query = '''
        INSERT INTO size_recommendations 
        (user_id, measurement_id, clothing_type, brand, recommended_size, 
         fit_score, details, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (recommendation.user_id, recommendation.measurement_id,
                  recommendation.clothing_type, recommendation.brand,
                  recommendation.recommended_size, recommendation.fit_score,
                  recommendation.details, recommendation.date)
                  
        recommendation_id = self._execute_insert(query, params)
        if recommendation_id > 0:
            recommendation.id = recommendation_id
        return recommendation_id
    
    def get_by_id(self, recommendation_id: int) -> Optional[SizeRecommendation]:
        query = "SELECT * FROM size_recommendations WHERE id = ?"
        params = (recommendation_id,)
        
        results = self._execute_query(query, params)
        if results:
            return SizeRecommendation.from_row(results[0])
        return None
    
    def get_for_user_and_type(
        self, 
        user_id: int, 
        clothing_type: str, 
        brand: Optional[str] = None
    ) -> List[SizeRecommendation]:
        if brand:
            query = '''
            SELECT * FROM size_recommendations 
            WHERE user_id = ? AND clothing_type = ? AND brand = ?
            ORDER BY date DESC
            '''
            params = (user_id, clothing_type, brand)
        else:
            query = '''
            SELECT * FROM size_recommendations 
            WHERE user_id = ? AND clothing_type = ?
            ORDER BY date DESC, brand
            '''
            params = (user_id, clothing_type)
        
        results = self._execute_query(query, params)
        return [SizeRecommendation.from_row(row) for row in results]
    
    def get_latest_for_user(
        self, 
        user_id: int, 
        limit: int = 10
    ) -> List[SizeRecommendation]:
        query = '''
        SELECT * FROM size_recommendations 
        WHERE user_id = ?
        ORDER BY date DESC
        LIMIT ?
        '''
        params = (user_id, limit)
        
        results = self._execute_query(query, params)
        return [SizeRecommendation.from_row(row) for row in results]
    
    def delete(self, recommendation_id: int) -> bool:
        query = "DELETE FROM size_recommendations WHERE id = ?"
        params = (recommendation_id,)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def delete_for_user_and_measurement(self, user_id: int, measurement_id: int) -> int:
        query = "DELETE FROM size_recommendations WHERE user_id = ? AND measurement_id = ?"
        params = (user_id, measurement_id)
        
        return self._execute_update(query, params)
    
    def get_size_distribution_by_brand(self, clothing_type: str) -> Dict[str, Dict[str, int]]:
        query = '''
        SELECT brand, recommended_size, COUNT(*) as count 
        FROM size_recommendations 
        WHERE clothing_type = ? 
        GROUP BY brand, recommended_size
        ORDER BY brand, count DESC
        '''
        params = (clothing_type,)
        
        results = self._execute_query(query, params)
        
        distribution = {}
        for row in results:
            brand = row["brand"]
            size = row["recommended_size"]
            count = row["count"]
            
            if brand not in distribution:
                distribution[brand] = {}
                
            distribution[brand][size] = count
            
        return distribution


# Funciones de conveniencia globales importadas por app.py
def save_measurement(user_id: int, measurements: dict, source_image_path: str) -> int:
    """Guarda una medición corporal en la base de datos."""
    repo = MeasurementRepository()
    m = Measurement(user_id=user_id)
    m.height = measurements.get("height")
    m.weight = measurements.get("weight")
    m.chest = measurements.get("chest")
    m.waist = measurements.get("waist")
    m.hips = measurements.get("hip") or measurements.get("hips")
    m.shoulders = measurements.get("shoulder_width") or measurements.get("shoulders")
    m.arm_length = measurements.get("arm_length")
    m.inseam = measurements.get("inseam")
    m.neck = measurements.get("neck")
    m.thigh = measurements.get("thigh")
    m.front_image = source_image_path
    
    # Rellenar landmarks si vienen en el dict
    if "landmarks" in measurements:
        m.landmarks = json.dumps(measurements["landmarks"])
        
    m.additional_measurements = json.dumps(measurements)
    return repo.create(m)


def get_user_measurements(measurement_id: int) -> Optional[Measurement]:
    """Obtiene una medición corporal por ID."""
    return MeasurementRepository().get_by_id(measurement_id)