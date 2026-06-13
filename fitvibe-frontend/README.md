# Frontend FitVibe (Aplicación SPA en Next.js)

Este directorio contiene el frontend de la plataforma **Probador Virtual de Ropa**, construido utilizando **Next.js 16**, **React 19**, **Tailwind CSS v4** y **Framer Motion**.

El frontend está estructurado como una Aplicación de Página Única (SPA) que se conecta con la API Flask local para el procesamiento de imágenes y la consulta del catálogo.

---

## 🛠️ Requisitos de Desarrollo

Asegúrate de contar con:
* **Node.js**: Versión 18.x o superior.
* **npm**: Gestor de paquetes oficial (incluido con Node.js).

---

## 🚀 Comandos de Desarrollo

Primero, instala todas las dependencias del proyecto:
```bash
npm install
```

### 1. Iniciar Servidor de Desarrollo
Para arrancar el frontend localmente con recarga en tiempo real:
```bash
npm run dev
```
La aplicación estará disponible en `http://localhost:3000`.

### 2. Compilar para Producción y Ofuscación
Para compilar la aplicación, exportar las páginas estáticas y ejecutar la protección del código JavaScript:
```bash
npm run build
```
* **¿Qué sucede al ejecutar `npm run build`?**:
  1. Ejecuta `next build`, el cual compila y exporta las vistas estáticas del frontend en el directorio `out/` (debido a la opción `output: 'export'` en `next.config.ts`).
  2. Ejecuta `node obfuscate.js`, un script que toma todos los archivos `.js` generados en la carpeta `out/` y les aplica técnicas avanzadas de ofuscación mediante `javascript-obfuscator` para proteger el código de producción.

---

## ⚙️ Configuración y Enrutamiento en Flask

Dado que el servidor Flask (`app.py`) en la raíz del proyecto es el encargado de servir la aplicación completa en producción:
* Los archivos compilados de Next.js se colocan en los directorios de archivos estáticos y plantillas del backend Flask para su despliegue final.
* La ruta `/_next/<path:path>` en Flask sirve los archivos estáticos de la interfaz compilada.
* Las peticiones dinámicas de vistas se sirven mediante `/templates/<filename>`.

---

## 📚 Tecnologías del Frontend

* **Framework**: [Next.js](https://nextjs.org/) (App Router).
* **Estilos**: [Tailwind CSS v4](https://tailwindcss.com/) y animaciones fluidas con [Framer Motion](https://motion.dev/).
* **Iconografía**: [Lucide React](https://lucide.dev/).
* **Ofuscador**: [JavaScript Obfuscator](https://github.com/javascript-obfuscator/javascript-obfuscator).
