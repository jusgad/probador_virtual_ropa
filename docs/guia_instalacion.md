# Guía de Instalación y Configuración del Sistema

Esta guía detalla los pasos para instalar, configurar y ejecutar el **Probador Virtual de Ropa** en sistemas Windows, Linux o macOS.

---

## 📋 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado lo siguiente en tu equipo:

1. **Python 3.10 o superior**: Requerido por MediaPipe, TensorFlow y Flask. Puedes verificar tu versión con:
   ```bash
   python --version
   ```
2. **Node.js (v18.x o superior) y npm**: Requerido para compilar y ejecutar el frontend en Next.js. Verifica con:
   ```bash
   node -v
   npm -v
   ```
3. **Cámara o Webcam (Opcional)**: Recomendada una resolución de al menos 720p para el uso del modo en vivo con óptimos resultados de detección corporal.

---

## 🛠️ Instalación del Backend (Python/Flask)

Sigue estos pasos en la raíz del proyecto para inicializar el backend de Python:

### 1. Clonar o Acceder al Proyecto
Abre tu terminal (PowerShell en Windows, o Bash en Linux/macOS) en el directorio raíz del proyecto.

### 2. Crear y Activar el Entorno Virtual
Se recomienda utilizar un entorno virtual (`venv`) para aislar las dependencias:

* **Crear el entorno virtual**:
  ```bash
  python -m venv mi_entorno
  ```
* **Activar el entorno virtual**:
  - **Windows (PowerShell)**:
    ```powershell
    mi_entorno\Scripts\activate
    ```
  - **Windows (CMD)**:
    ```cmd
    mi_entorno\Scripts\activate.bat
    ```
  - **Linux / macOS**:
    ```bash
    source mi_entorno/bin/activate
    ```

### 3. Instalar las Dependencias de Python
Instala todos los paquetes necesarios enumerados en `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Nota: Esto puede demorar varios minutos, ya que incluye librerías de visión artificial y aprendizaje automático como TensorFlow, OpenCV y MediaPipe.*

### 4. Configurar la Base de Datos SQLite
Inicializa el esquema de base de datos ejecutando el script de configuración:
```bash
python scripts/setup_db.py --reset --samples
```
* **Opciones disponibles**:
  - `--reset`: Limpia e inicializa las tablas desde cero (elimina datos previos).
  - `--samples`: Carga usuarios, prendas y tablas de tallaje de muestra ideales para pruebas iniciales.

### 5. Descargar los Modelos de Inteligencia Artificial
La aplicación requiere modelos pre-entrenados de MediaPipe y MoveNet para la segmentación y estimación de pose. Descárgalos con:
```bash
python scripts/download_models.py --all
```
* **Opciones disponibles**:
  - `--all`: Descarga los modelos de pose y segmentación (Recomendado).
  - `--pose`: Descarga únicamente el modelo de pose corporal.
  - `--segmentation`: Descarga únicamente el modelo de segmentación de silueta/ropa.
  - `--force`: Fuerza la descarga de los modelos incluso si ya existen localmente.

---

## 💻 Instalación y Compilación del Frontend (Next.js)

El frontend está desarrollado sobre Next.js y está ubicado en la carpeta `fitvibe-frontend/`.

### 1. Instalar dependencias de Node.js
Entra a la carpeta del frontend e instala los paquetes:
```bash
cd fitvibe-frontend
npm install
```

### 2. Ejecutar en Modo Desarrollo (Hot Reloading)
Para realizar modificaciones en el código de forma interactiva en el puerto 3000:
```bash
npm run dev
```

### 3. Compilar en Producción y Ofuscar Código
Para compilar y exportar el frontend estático para que Flask lo sirva directamente, ejecuta:
```bash
npm run build
```
*Este comando compilará el código de Next.js (`next build`), generará el build estático en la carpeta `out/` y posteriormente ejecutará el script `obfuscate.js` para proteger el código JavaScript en producción.*

---

## 🚀 Ejecución del Sistema completo

Una vez completada la instalación, puedes levantar la aplicación web ejecutando el backend en Flask:

1. Asegúrate de tener el entorno virtual activo.
2. Desde la raíz del proyecto, ejecuta:
   ```bash
   python app.py
   ```
3. La aplicación se iniciará en `http://127.0.0.1:5000/`. Abre este enlace en tu navegador web.

---

## ❓ Resolución de Problemas Comunes (Troubleshooting)

### Error: `ModuleNotFoundError: No module named 'mediapipe'`
* **Solución**: Asegúrate de haber activado el entorno virtual (`mi_entorno`) antes de instalar los requerimientos con `pip install -r requirements.txt`.

### Latencia alta o retraso al usar la Webcam en vivo
* **Solución**: MediaPipe puede consumir bastantes recursos de CPU. Asegúrate de cerrar aplicaciones pesadas en segundo plano. También puedes ajustar la resolución de la cámara en el archivo `config.json` (por ejemplo, a `640x480`).

### Error al conectar con la base de datos o tablas faltantes
* **Solución**: Si el archivo de base de datos se corrompe o no existe, ejecuta `python scripts/setup_db.py --reset --samples` para restablecer completamente la base de datos con los datos de prueba.
