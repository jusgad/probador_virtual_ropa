# Manual de Usuario y Guía de Operaciones

Esta guía te ayudará a interactuar con el sistema de Probador Virtual utilizando los recursos de interfaz gráfica (UI) expuestas en la zona web local.

## 1. El Proceso de Medición y Ajuste

El sistema de la app está basado en la interacción directa por lo que se priorizará un diseño responsivo. 

### Método A: Subida de Imágenes
1. Desde la página **Inicio** o **Dashboard**, haz clic en el botón *"Medición"*.
2. Elige el modo `Subir archivo` (Upload).
3. Selecciona una foto completa de cuerpo entero (bien iluminada; fondo preferentemente neutro, formato .JPG o .PNG).
4. El sistema encolará usando su backend UUID en menos de 2 segundos. Visualizarás tu fotograma de retorno lleno de `Landmarks` (puntos y trazos que denotarán tu geometría).

### Método B: Captura en Vivo (Webcam)
1. Navega hacia `"Prueba mediante Webcam"`. Otorga a tu navegador Web los permisos temporalmente requeridos.
2. Aléjate por lo menos a 1 metro  de la cámara para permitir a MediaPipe leer tus clavículas, cuellos y extensión de muslos/caderas. Haz clic en "Aceptar Medida". 
3. El frontend de Javascript consumirá la endpoint segura `/api/measure_webcam`, guardará un log de auditoría y procederás inmediatamente a los Resultados.

## 2. La Pantalla de Resultados y Compatibilidad Mundial

### Interpretación de Tu Talla a nivel Global
En base a los cálculos procesados de contorno de pecho y cintura (y de pierna, para pantalones), serás arrojado a un "Esquema Dinámico".

Verás dos pestañas: **Talla Principal (Para tu zona local/genérica)** y un apartado inferior llamado:
**"Tu Talla en el Mundo"**

Esta sección provee equivalencias. Ejemplo, quizás fuiste medido con una 34 en EEUU, simultáneamente se representará tu comparativo equivalente Europeo (`EU: 50`) y Asiático (`ASIA: XXL`), salvándote de errores si vas a comprar tus productos en internet u operadoras chinas / del viejo occidente.

### Confianza de la Inteligencia Artificial (Score de Ajuste)
Notarás porcentajes como `Ajuste al 85% (Holgado)`. No te asustes, la ropa por definición rara vez es "perfecta". El Probador usa tu longitud corporal frente al volumen de la ropa virtual impuesta para dictar una *"Puntuación"* basada fuertemente en si te apretará a nivel hombro/tórax, dándote hasta 2 recomendaciones extra (e.g. *Usar Talla Anterior si quieres porte entallado*).
