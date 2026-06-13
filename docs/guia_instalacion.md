# Guía de Instalación y Requisistos del Sistema

El Probador Virtual de Ropa es un sistema que hace un uso intensivo de bibliotecas de matemáticas (NumPy) y de visión en tiempo real. Configurar correctamente tu entorno garantizará que no sufras latencias a la hora de que el servidor Flask capture y deduzca modelos corporales 3D.

## Requisitos de Hardware (Sugeridos)
- **CPU**: Procesador de 4 núcleos o superior (Recomendable: Intel Core i5 / AMD Ryzen 5).
- **RAM**: 8GB Mínimo o de preferencia 16GB, debido a la carga temporal de MediaPipe en memoria.
- **Cámara**: Resolución recomendada 720p u óptima de 1080p, para que el analizador detecte hombros, cuellos y caderas a la perfección.

## Configuración Paso a Paso (Windows/Linux/Mac)

1. **Verificar clonado y versionado**
Asegúrate de contar con el código y tener Python instalado y activado en la variable de entorno, preferiblemente **Python 3.10 o superior**. Estando en la raíz del proyecto abre tu consola (Powershell o Bash).

2. **Crear e Inicializar Entorno Virtual (Venv)**
Recomendamos siempre aislar el proyecto de tus dependencias globales:
```bash
python -m venv mi_entorno
```
**Actívalo**:
* _Windows_: `mi_entorno\Scripts\activate`
* _Linux/macOS_: `source mi_entorno/bin/activate`

3. **Inyectar Requerimientos**
Ejecuta la instalación de la lista oficial de paquetes `requirements.txt`. El proceso demorará unos minutos ya que descargará TensorFlow OP, MediaPipe y Flask:
```bash
pip install -r requirements.txt
```

4. **Modelos Predictivos Base**
Algunos scripts bajo `/scripts` como la creación inicial de la base de datos y la descarga del modelo de poses precisará ejecutarse en limpio por primera vez:
```bash
python scripts/setup_db.py
python scripts/download_models.py
```

5. **Lanzar la Aplicación (Localhost)**
Si estás configurado correctamente, despliega mediante:
```bash
python app.py
```
Se anunciará que el servidor se encuentra corriendo en el puerto **5000** o **8080** dependiendo de la variante local. Abre tu explorador web e ingresa a `http://127.0.0.1:5000/`.
