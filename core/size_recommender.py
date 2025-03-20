"""
Módulo para la recomendación de tallas basada en medidas corporales.
Implementa algoritmos para determinar la talla ideal a partir de las
medidas del usuario y las tablas de tallas de diferentes marcas.
"""

import json
import os
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging


class SizeRecommender:
    """
    Clase para recomendar tallas de ropa.
    Utiliza las medidas del usuario y tablas de tallas para
    encontrar la mejor coincidencia.
    """

    def __init__(self, size_charts_directory: str = 'data/references/size_charts'):
        """
        Inicializa el recomendador de tallas.

        Args:
            size_charts_directory: Directorio con las tablas de tallas en formato JSON
        """
        self.size_charts_directory = size_charts_directory
        self.size_charts = self._load_size_charts()
        self.logger = logging.getLogger(__name__)
        
    def _load_size_charts(self) -> Dict[str, Dict[str, Any]]:
        """
        Carga las tablas de tallas desde archivos JSON.
        
        Returns:
            Diccionario con tablas de tallas para diferentes marcas y tipos de ropa
        """
        size_charts = {}
        
        try:
            # Asegurarse de que el directorio existe
            if not os.path.exists(self.size_charts_directory):
                self.logger.warning(f"El directorio {self.size_charts_directory} no existe")
                return size_charts
                
            # Recorrer archivos en el directorio
            for filename in os.listdir(self.size_charts_directory):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.size_charts_directory, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            # Intentar cargar el JSON
                            chart_data = json.load(file)
                            
                            # Extraer nombre de marca del nombre de archivo (example: brand_name.json)
                            brand_name = os.path.splitext(filename)[0]
                            
                            # Almacenar datos en el diccionario
                            size_charts[brand_name] = chart_data
                    except json.JSONDecodeError:
                        self.logger.error(f"Error al decodificar JSON en {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error al cargar tabla de tallas {file_path}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error al cargar tablas de tallas: {str(e)}")
            
        return size_charts
        
    def get_available_brands(self) -> List[str]:
        """
        Obtiene la lista de marcas disponibles.
        
        Returns:
            Lista de nombres de marcas
        """
        return list(self.size_charts.keys())
        
    def get_clothing_types_for_brand(self, brand: str) -> List[str]:
        """
        Obtiene los tipos de ropa disponibles para una marca.
        
        Args:
            brand: Nombre de la marca
            
        Returns:
            Lista de tipos de ropa
        """
        if brand in self.size_charts:
            return list(self.size_charts[brand].keys())
        return []
        
    def get_available_genders_for_brand_and_type(self, brand: str, clothing_type: str) -> List[str]:
        """
        Obtiene los géneros disponibles para una marca y tipo de ropa.
        
        Args:
            brand: Nombre de la marca
            clothing_type: Tipo de ropa (ej: "shirts", "pants")
            
        Returns:
            Lista de géneros (ej: "male", "female", "unisex")
        """
        if brand in self.size_charts and clothing_type in self.size_charts[brand]:
            return list(self.size_charts[brand][clothing_type].keys())
        return []
        
    def recommend_size(
        self, 
        measurements: Dict[str, float], 
        brand: str, 
        clothing_type: str, 
        gender: str = "unisex"
    ) -> Dict[str, Any]:
        """
        Recomienda la talla ideal basada en las medidas.
        
        Args:
            measurements: Diccionario con medidas corporales
            brand: Nombre de la marca
            clothing_type: Tipo de ropa ("shirts", "pants", etc.)
            gender: Género ("male", "female", "unisex")
            
        Returns:
            Diccionario con recomendación de talla y detalles
        """
        # Verificar que existan los datos de la marca y tipo de ropa
        if brand not in self.size_charts:
            return {"error": f"Marca '{brand}' no encontrada"}
            
        if clothing_type not in self.size_charts[brand]:
            return {"error": f"Tipo de ropa '{clothing_type}' no encontrado para la marca '{brand}'"}
            
        # Si el género especificado no existe, intentar usar unisex
        if gender not in self.size_charts[brand][clothing_type]:
            if "unisex" in self.size_charts[brand][clothing_type]:
                gender = "unisex"
            else:
                return {"error": f"Género '{gender}' no encontrado para {brand}/{clothing_type}"}
                
        # Obtener la tabla de tallas
        size_table = self.size_charts[brand][clothing_type][gender]
        
        # Llamar al método específico según el tipo de ropa
        if clothing_type == "shirts":
            return self._recommend_shirt_size(measurements, size_table)
        elif clothing_type == "pants":
            return self._recommend_pants_size(measurements, size_table)
        elif clothing_type == "dresses":
            return self._recommend_dress_size(measurements, size_table)
        else:
            # Método genérico para otros tipos de ropa
            return self._recommend_generic_size(measurements, size_table)
            
    def _recommend_shirt_size(self, measurements: Dict[str, float], size_table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recomienda talla de camisa/blusa.
        
        Args:
            measurements: Diccionario con medidas
            size_table: Tabla de tallas
            
        Returns:
            Resultado de la recomendación
        """
        # Medidas relevantes para camisas/blusas
        relevant_metrics = ["chest", "shoulder_width", "neck", "arm_length"]
        
        # Verificar que tenemos al menos algunas medidas relevantes
        available_metrics = [m for m in relevant_metrics if m in measurements]
        if not available_metrics:
            return {"error": "No hay medidas relevantes para recomendar talla de camisa"}
            
        # Ponderación de la importancia de cada medida
        metric_weights = {
            "chest": 0.5,           # El contorno de pecho es el más importante
            "shoulder_width": 0.25,  # Ancho de hombros
            "neck": 0.15,           # Contorno de cuello
            "arm_length": 0.1       # Longitud de brazo
        }
        
        # Normalizar pesos para las métricas disponibles
        total_weight = sum(metric_weights[m] for m in available_metrics)
        normalized_weights = {m: metric_weights[m] / total_weight for m in available_metrics}
        
        # Calcular compatibilidad para cada talla
        size_scores = {}
        size_details = {}
        
        for size_name, size_data in size_table.items():
            score = 0
            details = {}
            
            for metric in available_metrics:
                if f"{metric}_min" in size_data and f"{metric}_max" in size_data:
                    user_value = measurements[metric]
                    min_value = size_data[f"{metric}_min"]
                    max_value = size_data[f"{metric}_max"]
                    
                    # Calcular cuán bien encaja esta medida en el rango
                    # 1.0 = perfecto (en el centro del rango)
                    # 0.0 = fuera del rango
                    if user_value < min_value:
                        # Demasiado pequeño
                        fit_score = max(0, 1 - (min_value - user_value) / (min_value * 0.15))
                        fit_status = "small"
                    elif user_value > max_value:
                        # Demasiado grande
                        fit_score = max(0, 1 - (user_value - max_value) / (max_value * 0.15))
                        fit_status = "large"
                    else:
                        # Dentro del rango
                        # Mejor puntuación cuanto más cerca del centro
                        center = (min_value + max_value) / 2
                        normalized_distance = abs(user_value - center) / ((max_value - min_value) / 2)
                        fit_score = 1 - normalized_distance * 0.5  # Penalizar menos por estar dentro del rango
                        fit_status = "good"
                        
                    # Añadir a la puntuación total
                    weighted_score = fit_score * normalized_weights[metric]
                    score += weighted_score
                    
                    # Guardar detalles
                    details[metric] = {
                        "user_value": round(user_value, 1),
                        "min_value": min_value,
                        "max_value": max_value,
                        "fit_score": round(fit_score, 2),
                        "fit_status": fit_status
                    }
            
            # Guardar puntuación y detalles
            size_scores[size_name] = score
            size_details[size_name] = details
            
        # Encontrar la talla con mejor puntuación
        if not size_scores:
            return {"error": "No se pudo calcular ninguna puntuación de talla"}
            
        best_size = max(size_scores.items(), key=lambda x: x[1])
        best_size_name = best_size[0]
        best_size_score = best_size[1]
        
        # Clasificar el ajuste general
        fit_quality = "perfect"
        if best_size_score < 0.7:
            fit_quality = "poor"
        elif best_size_score < 0.85:
            fit_quality = "acceptable"
        elif best_size_score < 0.95:
            fit_quality = "good"
            
        # Preparar resultado
        result = {
            "recommended_size": best_size_name,
            "fit_score": round(best_size_score, 2),
            "fit_quality": fit_quality,
            "details": size_details[best_size_name],
            "all_sizes": {
                size: {
                    "score": round(score, 2),
                    "details": size_details[size]
                } for size, score in size_scores.items()
            }
        }
        
        # Añadir una recomendación textual
        tight_metrics = [m for m, d in size_details[best_size_name].items() 
                         if d["fit_status"] == "large" and d["fit_score"] < 0.8]
                         
        loose_metrics = [m for m, d in size_details[best_size_name].items() 
                         if d["fit_status"] == "small" and d["fit_score"] < 0.8]
                         
        if tight_metrics:
            result["recommendations"] = f"Esta talla puede quedar ajustada en: {', '.join(tight_metrics)}"
        elif loose_metrics:
            result["recommendations"] = f"Esta talla puede quedar holgada en: {', '.join(loose_metrics)}"
        
        return result
        
    def _recommend_pants_size(self, measurements: Dict[str, float], size_table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recomienda talla de pantalón.
        
        Args:
            measurements: Diccionario con medidas
            size_table: Tabla de tallas
            
        Returns:
            Resultado de la recomendación
        """
        # Medidas relevantes para pantalones
        relevant_metrics = ["waist", "hip", "inseam", "thigh"]
        
        # Verificar que tenemos al menos algunas medidas relevantes
        available_metrics = [m for m in relevant_metrics if m in measurements]
        if not available_metrics:
            return {"error": "No hay medidas relevantes para recomendar talla de pantalón"}
            
        # Ponderación de la importancia de cada medida
        metric_weights = {
            "waist": 0.4,   # Contorno de cintura
            "hip": 0.3,     # Contorno de cadera
            "inseam": 0.2,  # Largo de entrepierna
            "thigh": 0.1    # Contorno de muslo
        }
        
        # Normalizar pesos para las métricas disponibles
        total_weight = sum(metric_weights[m] for m in available_metrics)
        normalized_weights = {m: metric_weights[m] / total_weight for m in available_metrics}
        
        # Calcular compatibilidad para cada talla
        size_scores = {}
        size_details = {}
        
        for size_name, size_data in size_table.items():
            score = 0
            details = {}
            
            for metric in available_metrics:
                if f"{metric}_min" in size_data and f"{metric}_max" in size_data:
                    user_value = measurements[metric]
                    min_value = size_data[f"{metric}_min"]
                    max_value = size_data[f"{metric}_max"]
                    
                    # Calcular cuán bien encaja esta medida en el rango
                    if user_value < min_value:
                        # Demasiado pequeño
                        fit_score = max(0, 1 - (min_value - user_value) / (min_value * 0.15))
                        fit_status = "small"
                    elif user_value > max_value:
                        # Demasiado grande
                        fit_score = max(0, 1 - (user_value - max_value) / (max_value * 0.15))
                        fit_status = "large"
                    else:
                        # Dentro del rango
                        center = (min_value + max_value) / 2
                        normalized_distance = abs(user_value - center) / ((max_value - min_value) / 2)
                        fit_score = 1 - normalized_distance * 0.5
                        fit_status = "good"
                        
                    # Añadir a la puntuación total
                    weighted_score = fit_score * normalized_weights[metric]
                    score += weighted_score
                    
                    # Guardar detalles
                    details[metric] = {
                        "user_value": round(user_value, 1),
                        "min_value": min_value,
                        "max_value": max_value,
                        "fit_score": round(fit_score, 2),
                        "fit_status": fit_status
                    }
            
            # Guardar puntuación y detalles
            size_scores[size_name] = score
            size_details[size_name] = details
            
        # Encontrar la talla con mejor puntuación
        if not size_scores:
            return {"error": "No se pudo calcular ninguna puntuación de talla"}
            
        best_size = max(size_scores.items(), key=lambda x: x[1])
        best_size_name = best_size[0]
        best_size_score = best_size[1]
        
        # Clasificar el ajuste general
        fit_quality = "perfect"
        if best_size_score < 0.7:
            fit_quality = "poor"
        elif best_size_score < 0.85:
            fit_quality = "acceptable"
        elif best_size_score < 0.95:
            fit_quality = "good"
            
        # Preparar resultado
        result = {
            "recommended_size": best_size_name,
            "fit_score": round(best_size_score, 2),
            "fit_quality": fit_quality,
            "details": size_details[best_size_name],
            "all_sizes": {
                size: {
                    "score": round(score, 2),
                    "details": size_details[size]
                } for size, score in size_scores.items()
            }
        }
        
        # Añadir una recomendación textual
        tight_metrics = [self._format_metric_name(m) for m, d in size_details[best_size_name].items() 
                         if d["fit_status"] == "large" and d["fit_score"] < 0.8]
                         
        loose_metrics = [self._format_metric_name(m) for m, d in size_details[best_size_name].items() 
                         if d["fit_status"] == "small" and d["fit_score"] < 0.8]
                         
        if tight_metrics:
            result["recommendations"] = f"Esta talla puede quedar ajustada en: {', '.join(tight_metrics)}"
        elif loose_metrics:
            result["recommendations"] = f"Esta talla puede quedar holgada en: {', '.join(loose_metrics)}"
            
        # Detectar si necesita talla especial (largo/corto)
        if "inseam" in measurements and "inseam" in size_details[best_size_name]:
            inseam_detail = size_details[best_size_name]["inseam"]
            if inseam_detail["fit_status"] == "large" and inseam_detail["fit_score"] < 0.85:
                result["special_recommendation"] = "Considerar talla 'largo'"
            elif inseam_detail["fit_status"] == "small" and inseam_detail["fit_score"] < 0.85:
                result["special_recommendation"] = "Considerar talla 'corto'"
        
        return result
        
    def _recommend_dress_size(self, measurements: Dict[str, float], size_table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recomienda talla de vestido.
        
        Args:
            measurements: Diccionario con medidas
            size_table: Tabla de tallas
            
        Returns:
            Resultado de la recomendación
        """
        # Medidas relevantes para vestidos (combina camisa y falda)
        relevant_metrics = ["chest", "waist", "hip", "height"]
        
        # Verificar que tenemos al menos algunas medidas relevantes
        available_metrics = [m for m in relevant_metrics if m in measurements]
        if not available_metrics:
            return {"error": "No hay medidas relevantes para recomendar talla de vestido"}
            
        # Ponderación de la importancia de cada medida
        metric_weights = {
            "chest": 0.35,  # Contorno de pecho
            "waist": 0.3,   # Contorno de cintura
            "hip": 0.25,    # Contorno de cadera
            "height": 0.1   # Altura
        }
        
        # El resto es similar a los métodos anteriores...
        # Aplicamos la misma lógica de puntuación y selección
        
        # Normalizar pesos para las métricas disponibles
        total_weight = sum(metric_weights[m] for m in available_metrics)
        normalized_weights = {m: metric_weights[m] / total_weight for m in available_metrics}
        
        # Calcular compatibilidad para cada talla
        size_scores = {}
        size_details = {}
        
        for size_name, size_data in size_table.items():
            score = 0
            details = {}
            
            for metric in available_metrics:
                if f"{metric}_min" in size_data and f"{metric}_max" in size_data:
                    user_value = measurements[metric]
                    min_value = size_data[f"{metric}_min"]
                    max_value = size_data[f"{metric}_max"]
                    
                    # Calcular cuán bien encaja esta medida en el rango
                    if user_value < min_value:
                        fit_score = max(0, 1 - (min_value - user_value) / (min_value * 0.15))
                        fit_status = "small"
                    elif user_value > max_value:
                        fit_score = max(0, 1 - (user_value - max_value) / (max_value * 0.15))
                        fit_status = "large"
                    else:
                        center = (min_value + max_value) / 2
                        normalized_distance = abs(user_value - center) / ((max_value - min_value) / 2)
                        fit_score = 1 - normalized_distance * 0.5
                        fit_status = "good"
                        
                    weighted_score = fit_score * normalized_weights[metric]
                    score += weighted_score
                    
                    details[metric] = {
                        "user_value": round(user_value, 1),
                        "min_value": min_value,
                        "max_value": max_value,
                        "fit_score": round(fit_score, 2),
                        "fit_status": fit_status
                    }
            
            size_scores[size_name] = score
            size_details[size_name] = details
            
        # Encontrar la talla con mejor puntuación
        if not size_scores:
            return {"error": "No se pudo calcular ninguna puntuación de talla"}
            
        best_size = max(size_scores.items(), key=lambda x: x[1])
        best_size_name = best_size[0]
        best_size_score = best_size[1]
        
        # Resto del código similar a los métodos anteriores...
        fit_quality = "perfect"
        if best_size_score < 0.7:
            fit_quality = "poor"
        elif best_size_score < 0.85:
            fit_quality = "acceptable"
        elif best_size_score < 0.95:
            fit_quality = "good"
            
        result = {
            "recommended_size": best_size_name,
            "fit_score": round(best_size_score, 2),
            "fit_quality": fit_quality,
            "details": size_details[best_size_name],
            "all_sizes": {
                size: {
                    "score": round(score, 2),
                    "details": size_details[size]
                } for size, score in size_scores.items()
            }
        }
        
        return result
        
    def _recommend_generic_size(self, measurements: Dict[str, float], size_table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recomienda talla para cualquier tipo de prenda de forma genérica.
        
        Args:
            measurements: Diccionario con medidas
            size_table: Tabla de tallas
            
        Returns:
            Resultado de la recomendación
        """
        # Identificar qué métricas están presentes tanto en las medidas como en la tabla
        all_metrics = set()
        for size_data in size_table.values():
            for key in size_data.keys():
                if key.endswith("_min") or key.endswith("_max"):
                    metric = key[:-4]  # Quitar "_min" o "_max"
                    all_metrics.add(metric)
                    
        # Filtrar las métricas disponibles
        available_metrics = [m for m in all_metrics if m in measurements]
        
        if not available_metrics:
            return {"error": "No hay medidas coincidentes para recomendar una talla"}
            
        # Asignar pesos iguales a todas las métricas disponibles
        weight_per_metric = 1.0 / len(available_metrics)
        normalized_weights = {m: weight_per_metric for m in available_metrics}
        
        # El resto del método es similar a los anteriores...
        size_scores = {}
        size_details = {}
        
        for size_name, size_data in size_table.items():
            score = 0
            details = {}
            
            for metric in available_metrics:
                if f"{metric}_min" in size_data and f"{metric}_max" in size_data:
                    user_value = measurements[metric]
                    min_value = size_data[f"{metric}_min"]
                    max_value = size_data[f"{metric}_max"]
                    
                    if user_value < min_value:
                        fit_score = max(0, 1 - (min_value - user_value) / (min_value * 0.15))
                        fit_status = "small"
                    elif user_value > max_value:
                        fit_score = max(0, 1 - (user_value - max_value) / (max_value * 0.15))
                        fit_status = "large"
                    else:
                        center = (min_value + max_value) / 2
                        normalized_distance = abs(user_value - center) / ((max_value - min_value) / 2)
                        fit_score = 1 - normalized_distance * 0.5
                        fit_status = "good"
                        
                    weighted_score = fit_score * normalized_weights[metric]
                    score += weighted_score
                    
                    details[metric] = {
                        "user_value": round(user_value, 1),
                        "min_value": min_value,
                        "max_value": max_value,
                        "fit_score": round(fit_score, 2),
                        "fit_status": fit_status
                    }
            
            size_scores[size_name] = score
            size_details[size_name] = details
            
        # Encontrar la talla con mejor puntuación
        if not size_scores:
            return {"error": "No se pudo calcular ninguna puntuación de talla"}
            
        best_size = max(size_scores.items(), key=lambda x: x[1])
        best_size_name = best_size[0]
        best_size_score = best_size[1]
        
        # Clasificar el ajuste general
        fit_quality = "perfect"
        if best_size_score < 0.7:
            fit_quality = "poor"
        elif best_size_score < 0.85:
            fit_quality = "acceptable"
        elif best_size_score < 0.95:
            fit_quality = "good"
            
        result = {
            "recommended_size": best_size_name,
            "fit_score": round(best_size_score, 2),
            "fit_quality": fit_quality,
            "details": size_details[best_size_name],
            "all_sizes": {
                size: {
                    "score": round(score, 2)
                } for size, score in size_scores.items()
            }
        }
        
        return result
    
    def compare_brands(
        self, 
        measurements: Dict[str, float], 
        clothing_type: str, 
        brands: List[str], 
        gender: str = "unisex"
    ) -> Dict[str, Any]:
        """
        Compara la recomendación de talla entre diferentes marcas.
        
        Args:
            measurements: Diccionario con medidas corporales
            clothing_type: Tipo de ropa ("shirts", "pants", etc.)
            brands: Lista de marcas a comparar
            gender: Género ("male", "female", "unisex")
            
        Returns:
            Comparación de recomendaciones entre marcas
        """
        results = {}
        best_brands = []
        best_fit_score = 0
        
        for brand in brands:
            # Verificar que la marca existe y tiene el tipo de ropa
            if brand not in self.size_charts:
                results[brand] = {"error": f"Marca '{brand}' no encontrada"}
                continue
                
            if clothing_type not in self.size_charts[brand]:
                results[brand] = {"error": f"Tipo de ropa '{clothing_type}' no encontrado para la marca '{brand}'"}
                continue
            
            # Obtener recomendación para esta marca
            recommendation = self.recommend_size(measurements, brand, clothing_type, gender)
            results[brand] = recommendation
            
            # Actualizar mejor marca si corresponde
            if "fit_score" in recommendation and recommendation["fit_score"] > best_fit_score:
                best_fit_score = recommendation["fit_score"]
                best_brands = [brand]
            elif "fit_score" in recommendation and recommendation["fit_score"] == best_fit_score:
                best_brands.append(brand)
                
        # Añadir resumen de mejores marcas
        summary = {
            "best_brands": best_brands,
            "best_fit_score": best_fit_score,
            "brand_count": len(results)
        }
        
        return {
            "summary": summary,
            "brand_recommendations": results
        }
    
    def _format_metric_name(self, metric: str) -> str:
        """
        Formatea el nombre de una métrica para mostrar al usuario.
        
        Args:
            metric: Nombre interno de la métrica
            
        Returns:
            Nombre formateado para mostrar
        """
        name_map = {
            "chest": "contorno de pecho",
            "waist": "contorno de cintura",
            "hip": "contorno de cadera",
            "shoulder_width": "ancho de hombros",
            "arm_length": "longitud de brazo",
            "inseam": "entrepierna",
            "thigh": "contorno de muslo",
            "neck": "contorno de cuello",
            "height": "altura"
        }
        
        return name_map.get(metric, metric)