# Probador Virtual de Ropa (Virtual Fitting Room)

¡Bienvenido(a) al repositorio del **Probador Virtual de Ropa**! Esta plataforma utiliza visión artificial avanzada e inteligencia artificial para extraer medidas corporales precisas a partir de imágenes o de transmisiones en vivo vía cámara web, sugiriendo de forma inteligente la talla ideal en estándares de vestimenta de diferentes regiones del mundo.

---

## 🌟 Características Principales

* 🧠 **Visión Artificial con MediaPipe y OpenCV**: El sistema detecta y traza puntos de referencia corporales clave (hombros, cuello, pecho, cintura y caderas) para deducir dimensiones en centímetros o pulgadas de manera automática.
* 🌍 **Recomendación de Tallas Multiregional**: Compara de forma dinámica las mediciones del usuario con tablas de tallaje estandarizadas para distintas regiones, incluyendo:
  - Estados Unidos (US)
  - Europa (EU)
  - Reino Unido (UK)
  - Asia (ASIA)
  - Latinoamérica (LATAM)
* 🛡️ **Escudo de Seguridad y Anti-Cruces (Race Conditions)**: El backend desarrollado en Flask procesa y almacena temporalmente los archivos mediante identificadores únicos (`UUID`), aplicando validaciones estrictas de tipo de archivo para mitigar la inyección de archivos peligrosos y evitar colisiones de imágenes entre usuarios concurrentes.
* 💾 **Persistencia con Arquitectura SQL limpia**: Integra SQLite para el almacenamiento estructurado del historial del usuario (medidas y resultados de ajuste) mediante repositorios parametrizados y seguros.
* 💻 **Frontend Moderno e Interactivo (Single Page App)**: Una interfaz moderna creada con Next.js que proporciona una experiencia de usuario fluida, conectándose sin fricciones con la API de Flask.

---

## 📁 Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

* `core/`: Contiene la lógica del negocio y procesamiento de visión artificial.
  - `body_detector.py`: Módulo para la detección de siluetas y pose con MediaPipe.
  - `measurement.py`: Algoritmos para calcular las dimensiones anatómicas reales.
  - `clothing_fitter.py`: Lógica para simular cómo se ajusta la prenda sobre el cuerpo.
  - `size_recommender.py`: Motor de inferencia y recomendación de tallas.
* `database/`: Conexión de base de datos SQLite y repositorios parametrizados.
  - `db.py`: Conexiones thread-safe y definición de modelos/tablas.
  - `repositories.py`: Abstracción de acceso a datos de medición y prendas.
* `docs/`: Documentación detallada del proyecto (en español).
  - [guia_instalacion.md](file:///c:/Users/sebas/Documents/probador_virtual_ropa/docs/guia_instalacion.md): Pasos para configurar el backend de Python y el frontend de Next.js.
  - [manual_usuario.md](file:///c:/Users/sebas/Documents/probador_virtual_ropa/docs/manual_usuario.md): Manual para la operación y uso del probador virtual.
  - [arquitectura_api.md](file:///c:/Users/sebas/Documents/probador_virtual_ropa/docs/arquitectura_api.md): Detalles técnicos sobre la API Flask y el esquema de base de datos.
* `fitvibe-frontend/`: Proyecto frontend SPA construido sobre Next.js.
* `scripts/`: Utilidades administrativas.
  - `setup_db.py`: Configuración inicial del esquema de base de datos SQLite y carga de muestras.
  - `download_models.py`: Descarga y configuración automática de modelos de visión artificial pre-entrenados.
* `static/` y `templates/`: Archivos estáticos y vistas del backend Flask.
* `app.py`: Archivo de entrada de la aplicación Flask, exponiendo las APIs y las páginas.

---

## 🚀 Inicio Rápido

Para poner en marcha el sistema localmente:

1. **Configurar el entorno virtual de Python**:
   ```bash
   python -m venv mi_entorno
   # Activar en Windows:
   mi_entorno\Scripts\activate
   # Activar en Linux/macOS:
   source mi_entorno/bin/activate
   ```
2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Inicializar base de datos y descargar modelos de IA**:
   ```bash
   python scripts/setup_db.py --samples
   python scripts/download_models.py --all
   ```
4. **Ejecutar el backend Flask**:
   ```bash
   python app.py
   ```
5. **Ejecutar el frontend de desarrollo (Next.js)** (Opcional, en la carpeta `fitvibe-frontend/`):
   ```bash
   cd fitvibe-frontend
   npm install
   npm run dev
   ```

Para una guía paso a paso y más detalles sobre prerrequisitos, consulta la [Guía de Instalación](file:///c:/Users/sebas/Documents/probador_virtual_ropa/docs/guia_instalacion.md) en la carpeta `/docs`.

---

## 🛠️ Tecnologías Utilizadas

* **Backend**: Python 3.10+, Flask, SQLite3.
* **Procesamiento e IA**: OpenCV, MediaPipe Pose Landmarker, NumPy.
* **Frontend**: Next.js 16+, Framer Motion, Tailwind CSS, Lucide Icons.
