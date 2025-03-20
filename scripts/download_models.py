#!/usr/bin/env python3
"""
download_models.py - Script para descargar los modelos pre-entrenados necesarios para la aplicación.

Este script descarga modelos de pose estimation y segmentación requeridos por la aplicación
de prueba virtual de ropa. Coloca los modelos en los directorios adecuados y configura
los archivos JSON necesarios.

Uso:
    python scripts/download_models.py [--all] [--pose] [--segmentation] [--force]

Opciones:
    --all           Descarga todos los modelos disponibles
    --pose          Descarga solo modelos de pose estimation
    --segmentation  Descarga solo modelos de segmentación
    --force         Sobrescribe modelos existentes
"""

import os
import sys
import json
import argparse
import hashlib
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import urllib.request
import ssl

# Intentar importar tqdm para barras de progreso
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Agregar directorio raíz al path para importar módulos del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Directorios para modelos
MODELS_DIR = ROOT_DIR / "models"
POSE_MODELS_DIR = MODELS_DIR / "pose"
SEGMENTATION_MODELS_DIR = MODELS_DIR / "segmentation"

# URLs y MD5 para los modelos
MODEL_SOURCES = {
    "pose": [
        {
            "name": "MediaPipe Pose Landmarker",
            "filename": "mediapipe_pose_landmarker.task",
            "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            "md5": None,  # MD5 hash para verificación (opcional)
            "description": "Modelo de MediaPipe para detección de landmarks corporales en pose completa"
        },
        {
            "name": "PoseNet",
            "filename": "pose_estimation_model.pb",
            "url": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
            "md5": None,
            "description": "MoveNet Lightning, modelo ligero para estimación de pose humana"
        }
    ],
    "segmentation": [
        {
            "name": "Human Segmentation",
            "filename": "human_segmentation.pb",
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            "md5": None,
            "description": "Modelo de segmentación de personas (selfie segmenter de MediaPipe)"
        },
        {
            "name": "Clothing Segmentation",
            "filename": "clothing_segmentation.tflite",
            "url": "https://storage.googleapis.com/mediapipe-assets/object_segmenter.tflite",
            "md5": None,
            "description": "Modelo para segmentación de prendas de ropa"
        }
    ]
}

# Configuración para los modelos
MODEL_CONFIGS = {
    "pose": {
        "config_file": "pose_config.json",
        "config": {
            "mediapipe_pose_landmarker": {
                "model_path": "mediapipe_pose_landmarker.task",
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "output_segmentation_masks": False
            },
            "posenet": {
                "model_path": "pose_estimation_model.pb",
                "input_size": [192, 192],
                "output_stride": 16
            }
        }
    },
    "segmentation": {
        "config_file": "segmentation_config.json",
        "config": {
            "human_segmentation": {
                "model_path": "human_segmentation.pb",
                "input_size": [256, 256],
                "iou_threshold": 0.3
            },
            "clothing_segmentation": {
                "model_path": "clothing_segmentation.tflite",
                "input_size": [256, 256],
                "confidence_threshold": 0.5
            }
        }
    }
}


class DownloadProgressBar:
    """Clase para mostrar una barra de progreso durante las descargas."""
    
    def __init__(self, total=0, unit='B', unit_scale=True, desc=None):
        self.total = total
        self.desc = desc
        
        if TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                unit_scale=unit_scale,
                unit_divisor=1024,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        else:
            self.pbar = None
            self.last_percent = -1
            print(f"{desc}: 0%", end="", flush=True)
    
    def update(self, b=1):
        if self.pbar:
            self.pbar.update(b)
        else:
            if self.total > 0:
                percent = int((b / self.total) * 100)
                if percent > self.last_percent:
                    self.last_percent = percent
                    print(f"\r{self.desc}: {percent}%", end="", flush=True)
    
    def close(self):
        if self.pbar:
            self.pbar.close()
        else:
            print("\r{}: 100%".format(self.desc), flush=True)
            print()


def parse_args():
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Descarga modelos pre-entrenados para Virtual Fitting Room")
    parser.add_argument("--all", action="store_true", help="Descarga todos los modelos disponibles")
    parser.add_argument("--pose", action="store_true", help="Descarga solo modelos de pose estimation")
    parser.add_argument("--segmentation", action="store_true", help="Descarga solo modelos de segmentación")
    parser.add_argument("--force", action="store_true", help="Sobrescribe modelos existentes")
    parser.add_argument("--list", action="store_true", help="Lista los modelos disponibles sin descargar")
    
    args = parser.parse_args()
    
    # Si no se especifica ningún modelo, descargar todos
    if not (args.all or args.pose or args.segmentation or args.list):
        args.all = True
    
    return args


def ensure_directories():
    """Asegura que existan los directorios para modelos."""
    MODELS_DIR.mkdir(exist_ok=True)
    POSE_MODELS_DIR.mkdir(exist_ok=True)
    SEGMENTATION_MODELS_DIR.mkdir(exist_ok=True)


def verify_checksum(file_path: Path, expected_md5: Optional[str]) -> bool:
    """Verifica el checksum MD5 de un archivo."""
    if not expected_md5:
        return True  # Si no hay MD5 esperado, considerar válido
    
    print(f"Verificando checksum para {file_path}...")
    with open(file_path, "rb") as f:
        file_md5 = hashlib.md5(f.read()).hexdigest()
    
    if file_md5 != expected_md5:
        print(f"Error: Checksum no coincide para {file_path}")
        print(f"  Esperado: {expected_md5}")
        print(f"  Obtenido: {file_md5}")
        return False
    
    return True


def download_file(url: str, dest_path: Path, force: bool = False) -> bool:
    """Descarga un archivo desde una URL a una ruta destino."""
    if dest_path.exists() and not force:
        print(f"Archivo ya existe: {dest_path}")
        return True
    
    # Crear directorio si no existe
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Descargando {url} a {dest_path}...")
    
    # Para manejar certificados SSL en algunos entornos
    ssl_context = ssl._create_unverified_context()
    
    try:
        # Abrir conexión y obtener tamaño del archivo
        with urllib.request.urlopen(url, context=ssl_context) as response:
            file_size = int(response.info().get('Content-Length', 0))
            
            # Iniciar barra de progreso
            progress = DownloadProgressBar(
                total=file_size,
                desc=f"Descargando {dest_path.name}"
            )
            
            # Descargar archivo en bloques
            with open(dest_path, 'wb') as out_file:
                block_size = 8192
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    downloaded += len(buffer)
                    progress.update(len(buffer))
            
            progress.close()
        
        return True
    except Exception as e:
        print(f"Error al descargar {url}: {e}")
        # Eliminar archivo parcial si existe
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_archive(archive_path: Path, extract_dir: Path):
    """Extrae un archivo comprimido (zip, tar, tgz)."""
    print(f"Extrayendo {archive_path} a {extract_dir}...")
    
    # Crear directorio de extracción si no existe
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.suffix in ['.tar', '.tgz', '.gz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        print(f"Formato de archivo no soportado: {archive_path.suffix}")
        return False
    
    return True


def download_model(model_info: Dict, dest_dir: Path, force: bool = False) -> bool:
    """Descarga un modelo específico."""
    filename = model_info["filename"]
    url = model_info["url"]
    md5 = model_info["md5"]
    
    dest_path = dest_dir / filename
    
    if download_file(url, dest_path, force):
        if verify_checksum(dest_path, md5):
            print(f"Modelo descargado correctamente: {filename}")
            
            # Si es un archivo comprimido, extraerlo
            if filename.endswith(('.zip', '.tar', '.tgz', '.tar.gz')):
                extract_dir = dest_dir / filename.split('.')[0]
                if extract_archive(dest_path, extract_dir):
                    print(f"Archivo extraído correctamente en {extract_dir}")
            
            return True
    
    return False


def create_config_file(config_info: Dict, dest_dir: Path):
    """Crea un archivo de configuración JSON para los modelos."""
    config_path = dest_dir / config_info["config_file"]
    
    with open(config_path, 'w') as f:
        json.dump(config_info["config"], f, indent=2)
    
    print(f"Archivo de configuración creado: {config_path}")


def list_available_models():
    """Muestra una lista de los modelos disponibles para descarga."""
    print("\nModelos disponibles para descarga:\n")
    
    for category, models in MODEL_SOURCES.items():
        print(f"=== {category.upper()} ===")
        for model in models:
            print(f"  • {model['name']}")
            print(f"    Descripción: {model['description']}")
            print(f"    Archivo: {model['filename']}")
            print()


def download_models_by_category(category: str, force: bool = False):
    """Descarga todos los modelos de una categoría específica."""
    if category not in MODEL_SOURCES:
        print(f"Categoría desconocida: {category}")
        return
    
    models = MODEL_SOURCES[category]
    config = MODEL_CONFIGS[category]
    
    print(f"\nDescargando modelos de {category}...\n")
    
    # Determinar directorio destino
    if category == "pose":
        dest_dir = POSE_MODELS_DIR
    elif category == "segmentation":
        dest_dir = SEGMENTATION_MODELS_DIR
    else:
        dest_dir = MODELS_DIR / category
    
    # Asegurar que el directorio exista
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Descargar cada modelo
    success = True
    for model in models:
        print(f"\nDescargando {model['name']}...")
        if not download_model(model, dest_dir, force):
            success = False
            print(f"Error al descargar {model['name']}")
    
    # Crear archivo de configuración
    if success:
        create_config_file(config, dest_dir)
    
    return success


def main():
    """Función principal del script."""
    # Procesar argumentos
    args = parse_args()
    
    # Mostrar lista de modelos si se solicita
    if args.list:
        list_available_models()
        return
    
    # Asegurar que existan los directorios
    ensure_directories()
    
    # Descargar modelos según las opciones
    if args.all or args.pose:
        download_models_by_category("pose", args.force)
    
    if args.all or args.segmentation:
        download_models_by_category("segmentation", args.force)
    
    print("\nDescarga de modelos completada.")


if __name__ == "__main__":
    main()