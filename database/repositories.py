"""
Módulo de repositorios para el acceso a datos del sistema de prueba virtual.
Implementa el patrón repositorio para aislar la lógica de negocio del acceso a datos.
"""

import sqlite3
from typing import List, Dict, Optional, Any, Tuple, Union
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
        """
        Inicializa el repositorio base.
        """
        self.db = Database()
        self.logger = logging.getLogger(__name__)
        
    def _execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """
        Ejecuta una consulta SELECT y devuelve los resultados.
        
        Args:
            query: Consulta SQL
            params: Parámetros para la consulta
            
        Returns:
            Lista de diccionarios con los resultados
        """
        try:
            return self.db.execute_query(query, params)
        except Exception as e:
            self.logger.error(f"Error en consulta: {e}")
            return []
            
    def _execute_insert(self, query: str, params: tuple = ()) -> int:
        """
        Ejecuta una consulta INSERT y devuelve el ID generado.
        
        Args:
            query: Consulta SQL
            params: Parámetros para la consulta
            
        Returns:
            ID generado
        """
        try:
            return self.db.execute_insert(query, params)
        except Exception as e:
            self.logger.error(f"Error en inserción: {e}")
            return -1
            
    def _execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Ejecuta una consulta UPDATE/DELETE y devuelve el número de filas afectadas.
        
        Args:
            query: Consulta SQL
            params: Parámetros para la consulta
            
        Returns:
            Número de filas afectadas
        """
        try:
            return self.db.execute_update(query, params)
        except Exception as e:
            self.logger.error(f"Error en actualización: {e}")
            return 0


class UserRepository(BaseRepository):
    """
    Repositorio para operaciones relacionadas con usuarios.
    """
    
    def create(self, user: User) -> int:
        """
        Crea un nuevo usuario en la base de datos.
        
        Args:
            user: Objeto User a insertar
            
        Returns:
            ID del usuario creado o -1 si hay error
        """
        query = '''
        INSERT INTO users (email, password_hash, name, gender, birth_date, 
                          height, weight, created_at, last_login)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (user.email, user.password_hash, user.name, user.gender, 
                  user.birth_date, user.height, user.weight, user.created_at, 
                  user.last_login)
                  
        user_id = self._execute_insert(query, params)
        if user_id > 0:
            user.id = user_id
        return user_id
    
    def get_by_id(self, user_id: int) -> Optional[User]:
        """
        Obtiene un usuario por su ID.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Objeto User o None si no existe
        """
        query = "SELECT * FROM users WHERE id = ?"
        params = (user_id,)
        
        results = self._execute_query(query, params)
        if results:
            return User.from_row(results[0])
        return None
    
    def get_by_email(self, email: str) -> Optional[User]:
        """
        Obtiene un usuario por su email.
        
        Args:
            email: Email del usuario
            
        Returns:
            Objeto User o None si no existe
        """
        query = "SELECT * FROM users WHERE email = ?"
        params = (email,)
        
        results = self._execute_query(query, params)
        if results:
            return User.from_row(results[0])
        return None
    
    def update(self, user: User) -> bool:
        """
        Actualiza un usuario existente.
        
        Args:
            user: Objeto User con los datos actualizados
            
        Returns:
            True si la actualización fue exitosa
        """
        query = '''
        UPDATE users 
        SET email = ?, name = ?, gender = ?, birth_date = ?, 
            height = ?, weight = ?, last_login = ?
        WHERE id = ?
        '''
        params = (user.email, user.name, user.gender, user.birth_date,
                  user.height, user.weight, user.last_login, user.id)
                  
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def update_password(self, user_id: int, password_hash: str) -> bool:
        """
        Actualiza la contraseña de un usuario.
        
        Args:
            user_id: ID del usuario
            password_hash: Hash de la nueva contraseña
            
        Returns:
            True si la actualización fue exitosa
        """
        query = "UPDATE users SET password_hash = ? WHERE id = ?"
        params = (password_hash, user_id)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, user_id: int) -> bool:
        """
        Elimina un usuario por su ID.
        
        Args:
            user_id: ID del usuario a eliminar
            
        Returns:
            True si la eliminación fue exitosa
        """
        query = "DELETE FROM users WHERE id = ?"
        params = (user_id,)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[User]:
        """
        Obtiene una lista de usuarios con paginación.
        
        Args:
            limit: Límite de resultados
            offset: Desplazamiento para paginación
            
        Returns:
            Lista de objetos User
        """
        query = "SELECT * FROM users ORDER BY name LIMIT ? OFFSET ?"
        params = (limit, offset)
        
        results = self._execute_query(query, params)
        return [User.from_row(row) for row in results]
    
    def update_last_login(self, user_id: int) -> bool:
        """
        Actualiza la fecha del último inicio de sesión.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            True si la actualización fue exitosa
        """
        now = datetime.now().isoformat()
        query = "UPDATE users SET last_login = ? WHERE id = ?"
        params = (now, user_id)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0


class MeasurementRepository(BaseRepository):
    """
    Repositorio para operaciones relacionadas con medidas corporales.
    """
    
    def create(self, measurement: Measurement) -> int:
        """
        Crea un nuevo registro de medidas.
        
        Args:
            measurement: Objeto Measurement a insertar
            
        Returns:
            ID del registro creado o -1 si hay error
        """
        query = '''
        INSERT INTO measurements (user_id, date, height, weight, chest, waist, hip,
                                 shoulder_width, arm_length, inseam, neck, thigh, raw_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (measurement.user_id, measurement.date, measurement.height,
                  measurement.weight, measurement.chest, measurement.waist,
                  measurement.hip, measurement.shoulder_width, measurement.arm_length,
                  measurement.inseam, measurement.neck, measurement.thigh,
                  measurement.raw_data)
                  
        measurement_id = self._execute_insert(query, params)
        if measurement_id > 0:
            measurement.id = measurement_id
        return measurement_id
    
    def get_by_id(self, measurement_id: int) -> Optional[Measurement]:
        """
        Obtiene un registro de medidas por su ID.
        
        Args:
            measurement_id: ID del registro
            
        Returns:
            Objeto Measurement o None si no existe
        """
        query = "SELECT * FROM measurements WHERE id = ?"
        params = (measurement_id,)
        
        results = self._execute_query(query, params)
        if results:
            return Measurement.from_row(results[0])
        return None
    
    def get_latest_for_user(self, user_id: int) -> Optional[Measurement]:
        """
        Obtiene el registro de medidas más reciente para un usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Objeto Measurement más reciente o None
        """
        query = "SELECT * FROM measurements WHERE user_id = ? ORDER BY date DESC LIMIT 1"
        params = (user_id,)
        
        results = self._execute_query(query, params)
        if results:
            return Measurement.from_row(results[0])
        return None
    
    def get_all_for_user(self, user_id: int, limit: int = 10) -> List[Measurement]:
        """
        Obtiene todos los registros de medidas para un usuario.
        
        Args:
            user_id: ID del usuario
            limit: Límite de resultados
            
        Returns:
            Lista de objetos Measurement
        """
        query = "SELECT * FROM measurements WHERE user_id = ? ORDER BY date DESC LIMIT ?"
        params = (user_id, limit)
        
        results = self._execute_query(query, params)
        return [Measurement.from_row(row) for row in results]
    
    def update(self, measurement: Measurement) -> bool:
        """
        Actualiza un registro de medidas existente.
        
        Args:
            measurement: Objeto Measurement con datos actualizados
            
        Returns:
            True si la actualización fue exitosa
        """
        query = '''
        UPDATE measurements 
        SET height = ?, weight = ?, chest = ?, waist = ?, hip = ?,
            shoulder_width = ?, arm_length = ?, inseam = ?, neck = ?,
            thigh = ?, raw_data = ?
        WHERE id = ?
        '''
        params = (measurement.height, measurement.weight, measurement.chest,
                  measurement.waist, measurement.hip, measurement.shoulder_width,
                  measurement.arm_length, measurement.inseam, measurement.neck,
                  measurement.thigh, measurement.raw_data, measurement.id)
                  
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, measurement_id: int) -> bool:
        """
        Elimina un registro de medidas por su ID.
        
        Args:
            measurement_id: ID del registro a eliminar
            
        Returns:
            True si la eliminación fue exitosa
        """
        query = "DELETE FROM measurements WHERE id = ?"
        params = (measurement_id,)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def get_history(self, user_id: int, metric: str) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de una métrica específica para un usuario.
        
        Args:
            user_id: ID del usuario
            metric: Nombre de la métrica (chest, waist, etc.)
            
        Returns:
            Lista de diccionarios con fecha y valor
        """
        if metric not in ["height", "weight", "chest", "waist", "hip", 
                         "shoulder_width", "arm_length", "inseam", "neck", "thigh"]:
            self.logger.warning(f"Métrica inválida: {metric}")
            return []
            
        query = f"SELECT date, {metric} FROM measurements WHERE user_id = ? ORDER BY date"
        params = (user_id,)
        
        results = self._execute_query(query, params)
        return [{"date": row["date"], "value": row[metric]} for row in results if row[metric] is not None]


class ClothingRepository(BaseRepository):
    """
    Repositorio para operaciones relacionadas con prendas de ropa.
    """
    
    def create(self, clothing: ClothingItem) -> int:
        """
        Crea una nueva prenda en la base de datos.
        
        Args:
            clothing: Objeto ClothingItem a insertar
            
        Returns:
            ID de la prenda creada o -1 si hay error
        """
        query = '''
        INSERT INTO clothing_items (type, brand, name, size, color, image_path, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        params = (clothing.type, clothing.brand, clothing.name, clothing.size,
                  clothing.color, clothing.image_path, clothing.metadata)
                  
        clothing_id = self._execute_insert(query, params)
        if clothing_id > 0:
            clothing.id = clothing_id
        return clothing_id
    
    def get_by_id(self, clothing_id: int) -> Optional[ClothingItem]:
        """
        Obtiene una prenda por su ID.
        
        Args:
            clothing_id: ID de la prenda
            
        Returns:
            Objeto ClothingItem o None si no existe
        """
        query = "SELECT * FROM clothing_items WHERE id = ?"
        params = (clothing_id,)
        
        results = self._execute_query(query, params)
        if results:
            return ClothingItem.from_row(results[0])
        return None
    
    def get_by_type(self, clothing_type: str, limit: int = 50, offset: int = 0) -> List[ClothingItem]:
        """
        Obtiene prendas por tipo.
        
        Args:
            clothing_type: Tipo de prenda (shirt, pants, etc.)
            limit: Límite de resultados
            offset: Desplazamiento para paginación
            
        Returns:
            Lista de objetos ClothingItem
        """
        query = "SELECT * FROM clothing_items WHERE type = ? LIMIT ? OFFSET ?"
        params = (clothing_type, limit, offset)
        
        results = self._execute_query(query, params)
        return [ClothingItem.from_row(row) for row in results]
    
    def get_by_brand(self, brand: str, limit: int = 50, offset: int = 0) -> List[ClothingItem]:
        """
        Obtiene prendas por marca.
        
        Args:
            brand: Marca de ropa
            limit: Límite de resultados
            offset: Desplazamiento para paginación
            
        Returns:
            Lista de objetos ClothingItem
        """
        query = "SELECT * FROM clothing_items WHERE brand = ? LIMIT ? OFFSET ?"
        params = (brand, limit, offset)
        
        results = self._execute_query(query, params)
        return [ClothingItem.from_row(row) for row in results]
    
    def get_by_type_and_brand(self, clothing_type: str, brand: str) -> List[ClothingItem]:
        """
        Obtiene prendas por tipo y marca.
        
        Args:
            clothing_type: Tipo de prenda
            brand: Marca de ropa
            
        Returns:
            Lista de objetos ClothingItem
        """
        query = "SELECT * FROM clothing_items WHERE type = ? AND brand = ?"
        params = (clothing_type, brand)
        
        results = self._execute_query(query, params)
        return [ClothingItem.from_row(row) for row in results]
    
    def update(self, clothing: ClothingItem) -> bool:
        """
        Actualiza una prenda existente.
        
        Args:
            clothing: Objeto ClothingItem con datos actualizados
            
        Returns:
            True si la actualización fue exitosa
        """
        query = '''
        UPDATE clothing_items 
        SET type = ?, brand = ?, name = ?, size = ?, color = ?, 
            image_path = ?, metadata = ?
        WHERE id = ?
        '''
        params = (clothing.type, clothing.brand, clothing.name, clothing.size,
                  clothing.color, clothing.image_path, clothing.metadata, clothing.id)
                  
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, clothing_id: int) -> bool:
        """
        Elimina una prenda por su ID.
        
        Args:
            clothing_id: ID de la prenda a eliminar
            
        Returns:
            True si la eliminación fue exitosa
        """
        query = "DELETE FROM clothing_items WHERE id = ?"
        params = (clothing_id,)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def search(self, search_term: str, limit: int = 20) -> List[ClothingItem]:
        """
        Busca prendas por término de búsqueda.
        
        Args:
            search_term: Término a buscar
            limit: Límite de resultados
            
        Returns:
            Lista de objetos ClothingItem
        """
        # Búsqueda por marca, tipo, nombre o color
        query = '''
        SELECT * FROM clothing_items 
        WHERE brand LIKE ? OR type LIKE ? OR name LIKE ? OR color LIKE ?
        LIMIT ?
        '''
        search_pattern = f"%{search_term}%"
        params = (search_pattern, search_pattern, search_pattern, search_pattern, limit)
        
        results = self._execute_query(query, params)
        return [ClothingItem.from_row(row) for row in results]
    
    def get_available_brands(self) -> List[str]:
        """
        Obtiene todas las marcas disponibles.
        
        Returns:
            Lista de marcas únicas
        """
        query = "SELECT DISTINCT brand FROM clothing_items ORDER BY brand"
        
        results = self._execute_query(query)
        return [row["brand"] for row in results if row["brand"]]
    
    def get_available_types(self) -> List[str]:
        """
        Obtiene todos los tipos de prendas disponibles.
        
        Returns:
            Lista de tipos únicos
        """
        query = "SELECT DISTINCT type FROM clothing_items ORDER BY type"
        
        results = self._execute_query(query)
        return [row["type"] for row in results if row["type"]]
    
    def get_available_colors(self) -> List[str]:
        """
        Obtiene todos los colores disponibles.
        
        Returns:
            Lista de colores únicos
        """
        query = "SELECT DISTINCT color FROM clothing_items ORDER BY color"
        
        results = self._execute_query(query)
        return [row["color"] for row in results if row["color"]]


class FittingRepository(BaseRepository):
    """
    Repositorio para operaciones relacionadas con pruebas virtuales.
    """
    
    def create(self, fitting: VirtualFitting) -> int:
        """
        Crea una nueva prueba virtual.
        
        Args:
            fitting: Objeto VirtualFitting a insertar
            
        Returns:
            ID de la prueba creada o -1 si hay error
        """
        query = '''
        INSERT INTO virtual_fittings 
        (user_id, measurement_id, date, result_image_path, clothing_items, fit_scores, comments)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        params = (fitting.user_id, fitting.measurement_id, fitting.date,
                  fitting.result_image_path, fitting.clothing_items,
                  fitting.fit_scores, fitting.comments)
                  
        fitting_id = self._execute_insert(query, params)
        if fitting_id > 0:
            fitting.id = fitting_id
        return fitting_id
    
    def get_by_id(self, fitting_id: int) -> Optional[VirtualFitting]:
        """
        Obtiene una prueba virtual por su ID.
        
        Args:
            fitting_id: ID de la prueba
            
        Returns:
            Objeto VirtualFitting o None si no existe
        """
        query = "SELECT * FROM virtual_fittings WHERE id = ?"
        params = (fitting_id,)
        
        results = self._execute_query(query, params)
        if results:
            return VirtualFitting.from_row(results[0])
        return None
    
    def get_by_user(self, user_id: int, limit: int = 10) -> List[VirtualFitting]:
        """
        Obtiene pruebas virtuales para un usuario.
        
        Args:
            user_id: ID del usuario
            limit: Límite de resultados
            
        Returns:
            Lista de objetos VirtualFitting
        """
        query = "SELECT * FROM virtual_fittings WHERE user_id = ? ORDER BY date DESC LIMIT ?"
        params = (user_id, limit)
        
        results = self._execute_query(query, params)
        return [VirtualFitting.from_row(row) for row in results]
    
    def update(self, fitting: VirtualFitting) -> bool:
        """
        Actualiza una prueba virtual existente.
        
        Args:
            fitting: Objeto VirtualFitting con datos actualizados
            
        Returns:
            True si la actualización fue exitosa
        """
        query = '''
        UPDATE virtual_fittings 
        SET result_image_path = ?, clothing_items = ?, fit_scores = ?, comments = ?
        WHERE id = ?
        '''
        params = (fitting.result_image_path, fitting.clothing_items,
                  fitting.fit_scores, fitting.comments, fitting.id)
                  
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def delete(self, fitting_id: int) -> bool:
        """
        Elimina una prueba virtual por su ID.
        
        Args:
            fitting_id: ID de la prueba a eliminar
            
        Returns:
            True si la eliminación fue exitosa
        """
        query = "DELETE FROM virtual_fittings WHERE id = ?"
        params = (fitting_id,)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def get_user_history(self, user_id: int, full_details: bool = False) -> List[Dict[str, Any]]:
        """
        Obtiene el historial completo de pruebas para un usuario, con detalles adicionales.
        
        Args:
            user_id: ID del usuario
            full_details: Si se deben incluir detalles completos de prendas
            
        Returns:
            Lista de diccionarios con información de pruebas
        """
        if not full_details:
            # Versión simple
            query = "SELECT * FROM virtual_fittings WHERE user_id = ? ORDER BY date DESC"
            params = (user_id,)
            return self._execute_query(query, params)
        
        # Versión con detalles completos (incluye información de prendas)
        query = '''
        SELECT vf.*, m.date as measurement_date,
               GROUP_CONCAT(ci.id || '|' || ci.type || '|' || ci.brand || '|' || ci.name, ';') as clothing_details
        FROM virtual_fittings vf
        LEFT JOIN measurements m ON vf.measurement_id = m.id
        LEFT JOIN (
            SELECT id, type, brand, name 
            FROM clothing_items
        ) ci ON vf.clothing_items LIKE '%' || ci.id || '%'
        WHERE vf.user_id = ?
        GROUP BY vf.id
        ORDER BY vf.date DESC
        '''
        params = (user_id,)
        
        results = self._execute_query(query, params)
        
        # Procesamos los resultados para un formato más amigable
        processed_results = []
        for row in results:
            processed_row = dict(row)
            
            # Procesamos detalles de prendas
            if "clothing_details" in processed_row and processed_row["clothing_details"]:
                clothing_details = []
                for item in processed_row["clothing_details"].split(';'):
                    if item:
                        parts = item.split('|')
                        if len(parts) >= 4:
                            clothing_details.append({
                                "id": parts[0],
                                "type": parts[1],
                                "brand": parts[2],
                                "name": parts[3]
                            })
                processed_row["clothing_details"] = clothing_details
            else:
                processed_row["clothing_details"] = []
                
            processed_results.append(processed_row)
            
        return processed_results


class RecommendationRepository(BaseRepository):
    """
    Repositorio para operaciones relacionadas con recomendaciones de tallas.
    """
    
    def create(self, recommendation: SizeRecommendation) -> int:
        """
        Crea una nueva recomendación de talla.
        
        Args:
            recommendation: Objeto SizeRecommendation a insertar
            
        Returns:
            ID de la recomendación creada o -1 si hay error
        """
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
        """
        Obtiene una recomendación por su ID.
        
        Args:
            recommendation_id: ID de la recomendación
            
        Returns:
            Objeto SizeRecommendation o None si no existe
        """
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
        """
        Obtiene recomendaciones para un usuario y tipo de prenda.
        
        Args:
            user_id: ID del usuario
            clothing_type: Tipo de prenda
            brand: Marca (opcional)
            
        Returns:
            Lista de objetos SizeRecommendation
        """
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
        """
        Obtiene las recomendaciones más recientes para un usuario.
        
        Args:
            user_id: ID del usuario
            limit: Límite de resultados
            
        Returns:
            Lista de objetos SizeRecommendation
        """
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
        """
        Elimina una recomendación por su ID.
        
        Args:
            recommendation_id: ID de la recomendación a eliminar
            
        Returns:
            True si la eliminación fue exitosa
        """
        query = "DELETE FROM size_recommendations WHERE id = ?"
        params = (recommendation_id,)
        
        rows_affected = self._execute_update(query, params)
        return rows_affected > 0
    
    def delete_for_user_and_measurement(self, user_id: int, measurement_id: int) -> int:
        """
        Elimina recomendaciones para un usuario y medida específica.
        
        Args:
            user_id: ID del usuario
            measurement_id: ID de la medida
            
        Returns:
            Número de recomendaciones eliminadas
        """
        query = "DELETE FROM size_recommendations WHERE user_id = ? AND measurement_id = ?"
        params = (user_id, measurement_id)
        
        return self._execute_update(query, params)
    
    def get_size_distribution_by_brand(self, clothing_type: str) -> Dict[str, Dict[str, int]]:
        """
        Obtiene la distribución de tallas recomendadas por marca.
        
        Args:
            clothing_type: Tipo de prenda
            
        Returns:
            Diccionario con marcas y distribución de tallas
        """
        query = '''
        SELECT brand, recommended_size, COUNT(*) as count 
        FROM size_recommendations 
        WHERE clothing_type = ? 
        GROUP BY brand, recommended_size
        ORDER BY brand, count DESC
        '''
        params = (clothing_type,)
        
        results = self._execute_query(query, params)
        
        # Procesamos los resultados para obtener la distribución
        distribution = {}
        for row in results:
            brand = row["brand"]
            size = row["recommended_size"]
            count = row["count"]
            
            if brand not in distribution:
                distribution[brand] = {}
                
            distribution[brand][size] = count
            
        return distribution