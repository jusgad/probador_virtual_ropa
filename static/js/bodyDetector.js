/**
 * bodyDetector.js - Módulo para detección corporal y landmarks
 * 
 * Este módulo se encarga de:
 * - Inicializar el modelo de pose estimation de MediaPipe
 * - Gestionar el acceso a la cámara
 * - Detectar landmarks corporales en tiempo real o en imágenes
 * - Calcular distancias y proporciones entre landmarks
 */

/**
 * Clase principal para la detección corporal
 */
class BodyDetector {
    /**
     * Constructor
     * @param {Object} options - Opciones de configuración
     * @param {string} options.modelPath - Ruta al modelo de detección
     * @param {HTMLCanvasElement} options.canvas - Canvas para renderizado
     * @param {HTMLVideoElement} options.video - Elemento de video para la cámara
     */
    constructor(options = {}) {
        // Configuración
        this.modelPath = options.modelPath || '/models/pose/mediapipe_pose_landmarker.task';
        this.canvas = options.canvas || null;
        this.video = options.video || null;
        this.width = options.width || 640;
        this.height = options.height || 480;
        this.landmarks = null;
        this.detectionConfidence = 0.5;
        this.refreshRate = 30; // FPS
        
        // Estado
        this.isInitialized = false;
        this.isRunning = false;
        this.cameraActive = false;
        this.lastFrameTime = 0;
        
        // Configuración del modelo
        this.modelConfig = {
            baseOptions: {
                modelAssetPath: this.modelPath,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputSegmentationMasks: false
        };
        
        // MediaPipe
        this.poseDetector = null;
        this.animationFrame = null;
        
        // Contexto del canvas
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        
        // Callbacks
        this.onLandmarksDetected = null;
        
        // Constantes para los índices de landmarks (basados en MediaPipe Pose)
        this.LANDMARKS = {
            // Cara
            NOSE: 0,
            LEFT_EYE_INNER: 1,
            LEFT_EYE: 2,
            LEFT_EYE_OUTER: 3,
            RIGHT_EYE_INNER: 4,
            RIGHT_EYE: 5,
            RIGHT_EYE_OUTER: 6,
            LEFT_EAR: 7,
            RIGHT_EAR: 8,
            MOUTH_LEFT: 9,
            MOUTH_RIGHT: 10,
            
            // Hombros
            LEFT_SHOULDER: 11,
            RIGHT_SHOULDER: 12,
            
            // Codos
            LEFT_ELBOW: 13,
            RIGHT_ELBOW: 14,
            
            // Muñecas
            LEFT_WRIST: 15,
            RIGHT_WRIST: 16,
            
            // Manos
            LEFT_PINKY: 17,
            RIGHT_PINKY: 18,
            LEFT_INDEX: 19,
            RIGHT_INDEX: 20,
            LEFT_THUMB: 21,
            RIGHT_THUMB: 22,
            
            // Cadera
            LEFT_HIP: 23,
            RIGHT_HIP: 24,
            
            // Rodillas
            LEFT_KNEE: 25,
            RIGHT_KNEE: 26,
            
            // Tobillos
            LEFT_ANKLE: 27,
            RIGHT_ANKLE: 28,
            
            // Pies
            LEFT_HEEL: 29,
            RIGHT_HEEL: 30,
            LEFT_FOOT_INDEX: 31,
            RIGHT_FOOT_INDEX: 32
        };
    }
    
    /**
     * Inicializa el detector de pose
     * @returns {Promise<void>}
     */
    async initialize() {
        try {
            console.log("Inicializando detector corporal...");
            
            // Inicializar canvas si no está configurado
            if (this.canvas && this.ctx) {
                this.canvas.width = this.width;
                this.canvas.height = this.height;
            }
            
            // Cargar el modelo de postura desde MediaPipe
            const vision = await this.loadVisionPackage();
            
            // Crear el detector de pose
            this.poseDetector = await vision.PoseLandmarker.createFromOptions(
                this.modelConfig
            );
            
            this.isInitialized = true;
            console.log("Detector corporal inicializado correctamente");
            
            return true;
        } catch (error) {
            console.error("Error al inicializar el detector corporal:", error);
            throw error;
        }
    }
    
    /**
     * Carga el paquete de visión de MediaPipe
     * @returns {Promise<Object>} - El paquete de visión
     */
    async loadVisionPackage() {
        // Importar MediaPipe (asumiendo que está disponible globalmente o como módulo)
        if (window.FilesetResolver && window.PoseLandmarker) {
            const vision = {};
            
            // Crear el FilesetResolver
            vision.FilesetResolver = window.FilesetResolver;
            
            // Obtener el paquete de visión
            vision.PoseLandmarker = window.PoseLandmarker;
            
            return vision;
        } else {
            // Cargar dinámicamente si no está disponible
            try {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js';
                document.head.appendChild(script);
                
                return new Promise((resolve) => {
                    script.onload = () => {
                        const vision = {
                            FilesetResolver: window.FilesetResolver,
                            PoseLandmarker: window.PoseLandmarker
                        };
                        resolve(vision);
                    };
                });
            } catch (error) {
                console.error("Error al cargar el paquete de visión:", error);
                throw new Error("No se pudo cargar MediaPipe Vision");
            }
        }
    }
    
    /**
     * Inicia la detección en tiempo real usando la cámara
     * @returns {Promise<void>}
     */
    async startCamera() {
        if (!this.isInitialized) {
            throw new Error("El detector corporal no está inicializado");
        }
        
        if (this.cameraActive) {
            return; // Ya está activa
        }
        
        try {
            // Solicitar acceso a la cámara
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: this.width },
                    height: { ideal: this.height },
                    facingMode: "user"
                },
                audio: false
            });
            
            // Configurar el video
            if (this.video) {
                this.video.srcObject = stream;
                this.video.play();
                
                // Esperar a que el video esté listo
                await new Promise((resolve) => {
                    this.video.onloadedmetadata = () => {
                        // Actualizar dimensiones
                        this.width = this.video.videoWidth;
                        this.height = this.video.videoHeight;
                        
                        // Actualizar canvas
                        if (this.canvas) {
                            this.canvas.width = this.width;
                            this.canvas.height = this.height;
                        }
                        
                        resolve();
                    };
                });
                
                this.cameraActive = true;
                
                // Iniciar la detección
                this.startDetection();
                
                return true;
            } else {
                throw new Error("No se ha configurado un elemento de video");
            }
        } catch (error) {
            console.error("Error al iniciar la cámara:", error);
            throw error;
        }
    }
    
    /**
     * Detiene la cámara y la detección
     */
    stopCamera() {
        this.stopDetection();
        
        // Detener la cámara
        if (this.video && this.video.srcObject) {
            const tracks = this.video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            this.video.srcObject = null;
            this.cameraActive = false;
        }
    }
    
    /**
     * Inicia la detección continua
     */
    startDetection() {
        if (!this.isInitialized) {
            throw new Error("El detector corporal no está inicializado");
        }
        
        if (this.isRunning) {
            return; // Ya está en ejecución
        }
        
        this.isRunning = true;
        this.detectFrame();
    }
    
    /**
     * Detiene la detección continua
     */
    stopDetection() {
        this.isRunning = false;
        
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }
    
    /**
     * Detecta pose en un solo frame
     */
    async detectFrame() {
        if (!this.isRunning || !this.video) {
            return;
        }
        
        const now = performance.now();
        const elapsed = now - this.lastFrameTime;
        
        // Limitar la frecuencia de detección
        if (elapsed > 1000 / this.refreshRate) {
            this.lastFrameTime = now;
            
            try {
                // Realizar la detección
                if (this.video.readyState === 4) { // HAVE_ENOUGH_DATA
                    // Detectar pose
                    await this.detect(this.video);
                }
            } catch (error) {
                console.error("Error en la detección de pose:", error);
            }
        }
        
        // Programar el siguiente frame
        this.animationFrame = requestAnimationFrame(() => this.detectFrame());
    }
    
    /**
     * Realiza la detección en una imagen o video
     * @param {HTMLImageElement|HTMLVideoElement} input - Imagen o video para analizar
     * @returns {Promise<Object>} - Resultados de la detección
     */
    async detect(input) {
        if (!this.isInitialized) {
            throw new Error("El detector corporal no está inicializado");
        }
        
        try {
            // Realizar la detección
            let timestamp = 0;
            if (input instanceof HTMLVideoElement) {
                timestamp = input.currentTime * 1000; // ms
            }
            
            const results = await this.poseDetector.detectForVideo(input, timestamp);
            
            // Guardar los landmarks detectados
            if (results.landmarks && results.landmarks.length > 0) {
                this.landmarks = results.landmarks[0];
                
                // Dibujar landmarks si hay un canvas
                if (this.ctx && this.canvas) {
                    this.drawLandmarks();
                }
                
                // Llamar al callback si está definido
                if (typeof this.onLandmarksDetected === 'function') {
                    this.onLandmarksDetected(this.landmarks);
                }
            }
            
            return results;
        } catch (error) {
            console.error("Error en la detección:", error);
            throw error;
        }
    }
    
    /**
     * Dibuja los landmarks en el canvas
     */
    drawLandmarks() {
        if (!this.ctx || !this.canvas || !this.landmarks) {
            return;
        }
        
        // Limpiar el canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Voltear horizontalmente para modo espejo
        this.ctx.save();
        this.ctx.scale(-1, 1);
        this.ctx.translate(-this.canvas.width, 0);
        
        // Dibujar conexiones
        this.drawConnections();
        
        // Dibujar puntos
        this.landmarks.forEach((landmark, index) => {
            // Calcular coordenadas ajustadas al canvas
            const x = landmark.x * this.canvas.width;
            const y = landmark.y * this.canvas.height;
            
            // Tamaño según la visibilidad/confianza
            const pointSize = 4 + (landmark.visibility * 4);
            
            // Color según la parte del cuerpo
            let color = 'white';
            
            // Colorear por regiones
            if (index <= 10) {
                // Cara
                color = 'yellow';
            } else if (index <= 22) {
                // Brazos y hombros
                color = 'red';
            } else if (index <= 24) {
                // Cadera
                color = 'green';
            } else {
                // Piernas
                color = 'blue';
            }
            
            // Dibujar el punto
            this.ctx.beginPath();
            this.ctx.arc(x, y, pointSize, 0, 2 * Math.PI);
            this.ctx.fillStyle = color;
            this.ctx.fill();
            
            // Añadir índice para debug
            if (this.showIndices) {
                this.ctx.fillStyle = 'white';
                this.ctx.font = '10px Arial';
                this.ctx.fillText(index.toString(), x + 5, y - 5);
            }
        });
        
        this.ctx.restore();
    }
    
    /**
     * Dibuja las conexiones entre landmarks
     */
    drawConnections() {
        if (!this.ctx || !this.landmarks) {
            return;
        }
        
        // Definir las conexiones entre landmarks (basado en MediaPipe Pose)
        const connections = [
            // Cara
            [this.LANDMARKS.NOSE, this.LANDMARKS.LEFT_EYE_INNER],
            [this.LANDMARKS.LEFT_EYE_INNER, this.LANDMARKS.LEFT_EYE],
            [this.LANDMARKS.LEFT_EYE, this.LANDMARKS.LEFT_EYE_OUTER],
            [this.LANDMARKS.LEFT_EYE_OUTER, this.LANDMARKS.LEFT_EAR],
            [this.LANDMARKS.NOSE, this.LANDMARKS.RIGHT_EYE_INNER],
            [this.LANDMARKS.RIGHT_EYE_INNER, this.LANDMARKS.RIGHT_EYE],
            [this.LANDMARKS.RIGHT_EYE, this.LANDMARKS.RIGHT_EYE_OUTER],
            [this.LANDMARKS.RIGHT_EYE_OUTER, this.LANDMARKS.RIGHT_EAR],
            [this.LANDMARKS.MOUTH_LEFT, this.LANDMARKS.MOUTH_RIGHT],
            
            // Tronco
            [this.LANDMARKS.LEFT_SHOULDER, this.LANDMARKS.RIGHT_SHOULDER],
            [this.LANDMARKS.LEFT_SHOULDER, this.LANDMARKS.LEFT_HIP],
            [this.LANDMARKS.RIGHT_SHOULDER, this.LANDMARKS.RIGHT_HIP],
            [this.LANDMARKS.LEFT_HIP, this.LANDMARKS.RIGHT_HIP],
            
            // Brazo izquierdo
            [this.LANDMARKS.LEFT_SHOULDER, this.LANDMARKS.LEFT_ELBOW],
            [this.LANDMARKS.LEFT_ELBOW, this.LANDMARKS.LEFT_WRIST],
            [this.LANDMARKS.LEFT_WRIST, this.LANDMARKS.LEFT_PINKY],
            [this.LANDMARKS.LEFT_WRIST, this.LANDMARKS.LEFT_INDEX],
            [this.LANDMARKS.LEFT_WRIST, this.LANDMARKS.LEFT_THUMB],
            [this.LANDMARKS.LEFT_PINKY, this.LANDMARKS.LEFT_INDEX],
            
            // Brazo derecho
            [this.LANDMARKS.RIGHT_SHOULDER, this.LANDMARKS.RIGHT_ELBOW],
            [this.LANDMARKS.RIGHT_ELBOW, this.LANDMARKS.RIGHT_WRIST],
            [this.LANDMARKS.RIGHT_WRIST, this.LANDMARKS.RIGHT_PINKY],
            [this.LANDMARKS.RIGHT_WRIST, this.LANDMARKS.RIGHT_INDEX],
            [this.LANDMARKS.RIGHT_WRIST, this.LANDMARKS.RIGHT_THUMB],
            [this.LANDMARKS.RIGHT_PINKY, this.LANDMARKS.RIGHT_INDEX],
            
            // Pierna izquierda
            [this.LANDMARKS.LEFT_HIP, this.LANDMARKS.LEFT_KNEE],
            [this.LANDMARKS.LEFT_KNEE, this.LANDMARKS.LEFT_ANKLE],
            [this.LANDMARKS.LEFT_ANKLE, this.LANDMARKS.LEFT_HEEL],
            [this.LANDMARKS.LEFT_ANKLE, this.LANDMARKS.LEFT_FOOT_INDEX],
            [this.LANDMARKS.LEFT_HEEL, this.LANDMARKS.LEFT_FOOT_INDEX],
            
            // Pierna derecha
            [this.LANDMARKS.RIGHT_HIP, this.LANDMARKS.RIGHT_KNEE],
            [this.LANDMARKS.RIGHT_KNEE, this.LANDMARKS.RIGHT_ANKLE],
            [this.LANDMARKS.RIGHT_ANKLE, this.LANDMARKS.RIGHT_HEEL],
            [this.LANDMARKS.RIGHT_ANKLE, this.LANDMARKS.RIGHT_FOOT_INDEX],
            [this.LANDMARKS.RIGHT_HEEL, this.LANDMARKS.RIGHT_FOOT_INDEX]
        ];
        
        // Dibujar cada conexión
        connections.forEach(([start, end]) => {
            const startPoint = this.landmarks[start];
            const endPoint = this.landmarks[end];
            
            // Verificar visibilidad
            if (startPoint && endPoint && 
                startPoint.visibility > this.detectionConfidence && 
                endPoint.visibility > this.detectionConfidence) {
                
                const startX = startPoint.x * this.canvas.width;
                const startY = startPoint.y * this.canvas.height;
                const endX = endPoint.x * this.canvas.width;
                const endY = endPoint.y * this.canvas.height;
                
                this.ctx.beginPath();
                this.ctx.moveTo(startX, startY);
                this.ctx.lineTo(endX, endY);
                this.ctx.lineWidth = 2;
                this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
                this.ctx.stroke();
            }
        });
    }
    
    /**
     * Captura una imagen de la cámara actual
     * @returns {Promise<HTMLCanvasElement>} - Canvas con la imagen capturada
     */
    captureImage() {
        return new Promise((resolve, reject) => {
            if (!this.video || !this.cameraActive) {
                reject(new Error("La cámara no está activa"));
                return;
            }
            
            try {
                // Crear un canvas para la captura
                const captureCanvas = document.createElement('canvas');
                captureCanvas.width = this.width;
                captureCanvas.height = this.height;
                const captureCtx = captureCanvas.getContext('2d');
                
                // Dibujar el frame actual en modo espejo (como lo ve el usuario)
                captureCtx.save();
                captureCtx.scale(-1, 1);
                captureCtx.translate(-captureCanvas.width, 0);
                captureCtx.drawImage(this.video, 0, 0, captureCanvas.width, captureCanvas.height);
                
                // Dibujar los landmarks si están disponibles
                if (this.landmarks) {
                    // Reutilizamos la función pero con el contexto de captura
                    const originalCtx = this.ctx;
                    const originalCanvas = this.canvas;
                    
                    this.ctx = captureCtx;
                    this.canvas = captureCanvas;
                    this.drawConnections();
                    
                    // Restaurar el contexto original
                    this.ctx = originalCtx;
                    this.canvas = originalCanvas;
                }
                
                captureCtx.restore();
                
                resolve(captureCanvas);
            } catch (error) {
                reject(error);
            }
        });
    }
    
    /**
     * Selecciona una imagen desde el sistema de archivos
     * @returns {Promise<Object>} - Resultados de la detección
     */
    async selectImage() {
        return new Promise((resolve, reject) => {
            try {
                // Crear un input de archivo temporal
                const fileInput = document.createElement('input');
                fileInput.type = 'file';
                fileInput.accept = 'image/*';
                
                fileInput.onchange = async (e) => {
                    const file = e.target.files[0];
                    if (!file) {
                        reject(new Error("No se seleccionó ningún archivo"));
                        return;
                    }
                    
                    try {
                        // Convertir a imagen
                        const image = await this.fileToImage(file);
                        
                        // Detectar pose en la imagen
                        const results = await this.detectImage(image);
                        resolve(results);
                    } catch (error) {
                        reject(error);
                    }
                };
                
                // Simular clic para abrir el selector de archivos
                fileInput.click();
            } catch (error) {
                reject(error);
            }
        });
    }
    
    /**
     * Convierte un archivo a imagen
     * @param {File} file - Archivo de imagen
     * @returns {Promise<HTMLImageElement>} - Elemento de imagen
     */
    fileToImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                const img = new Image();
                
                img.onload = () => {
                    resolve(img);
                };
                
                img.onerror = () => {
                    reject(new Error("No se pudo cargar la imagen"));
                };
                
                img.src = e.target.result;
            };
            
            reader.onerror = () => {
                reject(new Error("Error al leer el archivo"));
            };
            
            reader.readAsDataURL(file);
        });
    }
    
    /**
     * Detecta pose en una imagen estática
     * @param {HTMLImageElement} image - Imagen para analizar
     * @returns {Promise<Object>} - Resultados de la detección con la imagen
     */
    async detectImage(image) {
        if (!this.isInitialized) {
            throw new Error("El detector corporal no está inicializado");
        }
        
        try {
            // Preparar el canvas para la imagen
            const imgCanvas = document.createElement('canvas');
            const imgCtx = imgCanvas.getContext('2d');
            
            // Ajustar tamaño del canvas a la imagen
            imgCanvas.width = image.width;
            imgCanvas.height = image.height;
            
            // Dibujar la imagen en el canvas
            imgCtx.drawImage(image, 0, 0);
            
            // Detectar pose en la imagen
            const results = await this.detect(image);
            
            // Dibujar landmarks en el canvas de imagen
            if (results.landmarks && results.landmarks.length > 0) {
                this.landmarks = results.landmarks[0];
                
                const originalCtx = this.ctx;
                const originalCanvas = this.canvas;
                const originalWidth = this.width;
                const originalHeight = this.height;
                
                // Usar el canvas de la imagen temporalmente
                this.ctx = imgCtx;
                this.canvas = imgCanvas;
                this.width = image.width;
                this.height = image.height;
                
                // Dibujar landmarks
                this.drawLandmarks();
                
                // Restaurar valores originales
                this.ctx = originalCtx;
                this.canvas = originalCanvas;
                this.width = originalWidth;
                this.height = originalHeight;
            }
            
            return {
                results: results,
                image: imgCanvas
            };
        } catch (error) {
            console.error("Error en detectImage:", error);
            throw error;
        }
    }
    
    /**
     * Calcula la distancia entre dos landmarks
     * @param {number} landmarkIndex1 - Índice del primer landmark
     * @param {number} landmarkIndex2 - Índice del segundo landmark
     * @returns {number} - Distancia normalizada (0-1)
     */
    getDistance(landmarkIndex1, landmarkIndex2) {
        if (!this.landmarks) {
            return 0;
        }
        
        const landmark1 = this.landmarks[landmarkIndex1];
        const landmark2 = this.landmarks[landmarkIndex2];
        
        if (!landmark1 || !landmark2) {
            return 0;
        }
        
        // Distancia euclidiana 3D
        const dx = landmark2.x - landmark1.x;
        const dy = landmark2.y - landmark1.y;
        const dz = landmark2.z - landmark1.z;
        
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    
    /**
     * Calcula la distancia en píxeles entre dos landmarks
     * @param {number} landmarkIndex1 - Índice del primer landmark
     * @param {number} landmarkIndex2 - Índice del segundo landmark
     * @returns {number} - Distancia en píxeles
     */
    getDistanceInPixels(landmarkIndex1, landmarkIndex2) {
        if (!this.landmarks || !this.canvas) {
            return 0;
        }
        
        const landmark1 = this.landmarks[landmarkIndex1];
        const landmark2 = this.landmarks[landmarkIndex2];
        
        if (!landmark1 || !landmark2) {
            return 0;
        }
        
        // Calcular coordenadas en píxeles
        const x1 = landmark1.x * this.canvas.width;
        const y1 = landmark1.y * this.canvas.height;
        const x2 = landmark2.x * this.canvas.width;
        const y2 = landmark2.y * this.canvas.height;
        
        // Distancia euclidiana 2D
        const dx = x2 - x1;
        const dy = y2 - y1;
        
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    /**
     * Obtiene las dimensiones del cuerpo basadas en landmarks
     * @returns {Object} - Dimensiones corporales
     */
    getBodyDimensions() {
        if (!this.landmarks) {
            return null;
        }
        
        // Calcular dimensiones de interés para vestimenta
        const dimensions = {
            // Anchura de hombros
            shoulderWidth: this.getDistanceInPixels(
                this.LANDMARKS.LEFT_SHOULDER,
                this.LANDMARKS.RIGHT_SHOULDER
            ),
            
            // Anchura de cadera
            hipWidth: this.getDistanceInPixels(
                this.LANDMARKS.LEFT_HIP,
                this.LANDMARKS.RIGHT_HIP
            ),
            
            // Longitud del torso
            torsoLength: (
                this.getDistanceInPixels(
                    this.LANDMARKS.LEFT_SHOULDER,
                    this.LANDMARKS.LEFT_HIP
                ) +
                this.getDistanceInPixels(
                    this.LANDMARKS.RIGHT_SHOULDER,
                    this.LANDMARKS.RIGHT_HIP
                )
            ) / 2,
            
            // Longitud de brazo
            armLength: (
                this.getDistanceInPixels(
                    this.LANDMARKS.LEFT_SHOULDER,
                    this.LANDMARKS.LEFT_ELBOW
                ) +
                this.getDistanceInPixels(
                    this.LANDMARKS.LEFT_ELBOW,
                    this.LANDMARKS.LEFT_WRIST
                )
            ),
            
            // Longitud de pierna
            legLength: (
                this.getDistanceInPixels(
                    this.LANDMARKS.LEFT_HIP,
                    this.LANDMARKS.LEFT_KNEE
                ) +
                this.getDistanceInPixels(
                    this.LANDMARKS.LEFT_KNEE,
                    this.LANDMARKS.LEFT_ANKLE
                )
            ),
            
            // Altura total (aproximada)
            height: this.getDistanceInPixels(
                this.LANDMARKS.NOSE,
                this.LANDMARKS.LEFT_ANKLE
            ),
            
            // Altura entre cadera y cuello (para camisas)
            shirtLength: this.getDistanceInPixels(
                this.LANDMARKS.LEFT_HIP,
                this.LANDMARKS.LEFT_SHOULDER
            ),
            
            // Distancia entre codos (aproximación de pecho)
            chestWidth: this.getDistanceInPixels(
                this.LANDMARKS.LEFT_ELBOW,
                this.LANDMARKS.RIGHT_ELBOW
            )
        };
        
        return dimensions;
    }
    
    /**
     * Configura un callback para cuando se detectan landmarks
     * @param {Function} callback - Función a llamar con los landmarks
     */
    setLandmarksCallback(callback) {
        if (typeof callback === 'function') {
            this.onLandmarksDetected = callback;
        }
    }
    
    /**
     * Limpia recursos y cierra el detector
     */
    dispose() {
        this.stopCamera();
        
        if (this.poseDetector) {
            this.poseDetector.close();
            this.poseDetector = null;
        }
        
        this.isInitialized = false;
    }
}

// Exportar la clase para uso en otros módulos
export default BodyDetector;