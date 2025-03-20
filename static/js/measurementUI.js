/**
 * measurementUI.js - Interfaz de usuario para la toma de medidas
 * 
 * Este módulo se encarga de:
 * - Mostrar la interfaz para captura de imágenes/video
 * - Proporcionar instrucciones al usuario sobre cómo posicionarse
 * - Visualizar resultados de medición
 * - Gestionar el flujo del proceso de medición
 */

/**
 * Clase principal para la interfaz de medición
 */
class MeasurementUI {
    /**
     * Constructor
     * @param {Object} options - Opciones de configuración
     * @param {HTMLElement} options.resultsContainer - Contenedor para mostrar resultados
     * @param {HTMLElement} options.captureButton - Botón para capturar imagen
     * @param {HTMLElement} options.uploadButton - Botón para subir imagen
     * @param {HTMLElement} options.instructionsContainer - Contenedor para instrucciones
     * @param {Object} options.bodyDetector - Instancia de BodyDetector
     */
    constructor(options = {}) {
        // Elementos DOM
        this.resultsContainer = options.resultsContainer || null;
        this.captureButton = options.captureButton || null;
        this.uploadButton = options.uploadButton || null;
        this.instructionsContainer = options.instructionsContainer || null;
        
        // Referencias a módulos
        this.bodyDetector = options.bodyDetector || null;
        
        // Estado
        this.isSetup = false;
        this.currentStep = 0;
        this.measurements = null;
        this.measurementInProgress = false;
        this.lastImageCapture = null;
        
        // Constantes
        this.UNIT = 'cm'; // Unidad de medida (cm o pulgadas)
        this.STEPS = [
            {
                id: 'preparation',
                title: 'Preparación',
                instructions: [
                    'Usa ropa ajustada o deportiva para obtener medidas más precisas.',
                    'Párate derecho frente a la cámara, a unos 2 metros de distancia.',
                    'Asegúrate de que todo tu cuerpo sea visible en la imagen.',
                    'El lugar debe estar bien iluminado, preferiblemente con luz natural.'
                ]
            },
            {
                id: 'frontpose',
                title: 'Posición frontal',
                instructions: [
                    'Párate derecho con los pies separados al ancho de los hombros.',
                    'Brazos ligeramente separados del cuerpo (unos 15-20 cm).',
                    'Mira directamente a la cámara.',
                    'Mantén una postura natural, sin contraer músculos.'
                ],
                poseImage: '/static/img/poses/front_pose.png'
            },
            {
                id: 'sidepose',
                title: 'Posición lateral',
                instructions: [
                    'Gira 90 grados hacia tu derecha.',
                    'Mantén los brazos relajados a los costados.',
                    'Mira al frente, no a la cámara.',
                    'Mantén una postura recta y natural.'
                ],
                poseImage: '/static/img/poses/side_pose.png'
            },
            {
                id: 'results',
                title: 'Resultados',
                instructions: [
                    'Revisa las medidas calculadas.',
                    'Puedes editar manualmente cualquier medida si es necesario.',
                    'Guarda tus medidas para usarlas en pruebas virtuales de ropa.'
                ]
            }
        ];
        
        // Referencias a elementos creados dinámicamente
        this.progressIndicator = null;
        this.stepTitle = null;
        this.instructionsList = null;
        this.poseImage = null;
        this.alertMessage = null;
        this.measurementGrid = null;
        this.actionButtons = null;
        
        // Callbacks
        this.onMeasurementsComplete = null;
        this.onMeasurementsUpdated = null;
    }
    
    /**
     * Configura la interfaz de medición
     * @returns {boolean} - Éxito de la configuración
     */
    setup() {
        if (this.isSetup) {
            return true;
        }
        
        try {
            console.log("Configurando interfaz de medición...");
            
            // Verificar elementos necesarios
            if (!this.resultsContainer) {
                console.error("No se ha proporcionado un contenedor para resultados");
                return false;
            }
            
            // Crear estructura de UI
            this.createUI();
            
            // Configurar eventos
            this.setupEventListeners();
            
            // Iniciar en el primer paso
            this.goToStep(0);
            
            this.isSetup = true;
            console.log("Interfaz de medición configurada correctamente");
            
            return true;
        } catch (error) {
            console.error("Error al configurar la interfaz de medición:", error);
            return false;
        }
    }
    
    /**
     * Crea la estructura de UI
     */
    createUI() {
        // Crear indicador de progreso
        this.createProgressIndicator();
        
        // Crear contenedor de instrucciones si no existe
        if (!this.instructionsContainer) {
            this.instructionsContainer = document.createElement('div');
            this.instructionsContainer.className = 'measurement-instructions';
            document.querySelector('.measurement-container').appendChild(this.instructionsContainer);
        }
        
        // Crear elementos dentro del contenedor de instrucciones
        this.stepTitle = document.createElement('h3');
        this.stepTitle.className = 'instructions-title';
        
        this.instructionsList = document.createElement('ul');
        this.instructionsList.className = 'instructions-list';
        
        this.poseImage = document.createElement('img');
        this.poseImage.className = 'pose-reference';
        this.poseImage.style.display = 'none';
        
        this.alertMessage = document.createElement('div');
        this.alertMessage.className = 'alert-message';
        this.alertMessage.style.display = 'none';
        
        this.instructionsContainer.appendChild(this.stepTitle);
        this.instructionsContainer.appendChild(this.instructionsList);
        this.instructionsContainer.appendChild(this.poseImage);
        this.instructionsContainer.appendChild(this.alertMessage);
        
        // Crear grid de resultados de medición
        this.createMeasurementGrid();
        
        // Crear botones de acción
        this.createActionButtons();
    }
    
    /**
     * Crea el indicador de progreso
     */
    createProgressIndicator() {
        // Contenedor principal
        this.progressIndicator = document.createElement('div');
        this.progressIndicator.className = 'measurement-progress';
        
        // Línea de progreso base
        const progressLine = document.createElement('div');
        progressLine.className = 'progress-line';
        this.progressIndicator.appendChild(progressLine);
        
        // Línea de progreso rellena (dinámica)
        this.progressLineFilled = document.createElement('div');
        this.progressLineFilled.className = 'progress-line-filled';
        this.progressIndicator.appendChild(this.progressLineFilled);
        
        // Crear paso para cada etapa
        this.STEPS.forEach((step, index) => {
            const stepElement = document.createElement('div');
            stepElement.className = 'progress-step';
            stepElement.setAttribute('data-step', index);
            
            const stepNumber = document.createElement('div');
            stepNumber.className = 'progress-step-number';
            stepNumber.textContent = index + 1;
            
            const stepText = document.createElement('div');
            stepText.className = 'progress-step-text';
            stepText.textContent = step.title;
            
            stepElement.appendChild(stepNumber);
            stepElement.appendChild(stepText);
            
            // Añadir evento click para navegar a ese paso
            stepElement.addEventListener('click', () => {
                // Solo permitir navegar a pasos ya completados o el siguiente
                if (index <= this.currentStep + 1) {
                    this.goToStep(index);
                }
            });
            
            this.progressIndicator.appendChild(stepElement);
        });
        
        // Insertar al principio del contenedor principal
        const container = document.querySelector('.measurement-container');
        container.insertBefore(this.progressIndicator, container.firstChild);
    }
    
    /**
     * Crea la grilla de resultados de medición
     */
    createMeasurementGrid() {
        // Contenedor principal
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'measurement-results';
        
        // Título
        const resultsTitle = document.createElement('h3');
        resultsTitle.className = 'results-title';
        resultsTitle.textContent = 'Tus medidas';
        resultsContainer.appendChild(resultsTitle);
        
        // Grid para medidas
        this.measurementGrid = document.createElement('div');
        this.measurementGrid.className = 'results-grid';
        resultsContainer.appendChild(this.measurementGrid);
        
        // Añadir al contenedor de resultados
        if (this.resultsContainer) {
            this.resultsContainer.appendChild(resultsContainer);
        } else {
            document.querySelector('.measurement-container').appendChild(resultsContainer);
        }
    }
    
    /**
     * Crea los botones de acción
     */
    createActionButtons() {
        // Contenedor de botones
        this.actionButtons = document.createElement('div');
        this.actionButtons.className = 'action-buttons';
        
        // Botón anterior
        this.prevButton = document.createElement('button');
        this.prevButton.className = 'action-button secondary-action prev-button';
        this.prevButton.textContent = 'Anterior';
        this.prevButton.addEventListener('click', () => {
            this.goToPreviousStep();
        });
        
        // Botón siguiente
        this.nextButton = document.createElement('button');
        this.nextButton.className = 'action-button primary-action next-button';
        this.nextButton.textContent = 'Siguiente';
        this.nextButton.addEventListener('click', () => {
            this.goToNextStep();
        });
        
        // Botón guardar
        this.saveButton = document.createElement('button');
        this.saveButton.className = 'action-button primary-action save-button';
        this.saveButton.textContent = 'Guardar medidas';
        this.saveButton.style.display = 'none';
        this.saveButton.addEventListener('click', () => {
            this.saveMeasurements();
        });
        
        // Añadir botones al contenedor
        this.actionButtons.appendChild(this.prevButton);
        this.actionButtons.appendChild(this.nextButton);
        this.actionButtons.appendChild(this.saveButton);
        
        // Añadir al final del contenedor principal
        document.querySelector('.measurement-container').appendChild(this.actionButtons);
    }
    
    /**
     * Configura los event listeners
     */
    setupEventListeners() {
        // Configurar botón de captura si existe
        if (this.captureButton) {
            this.captureButton.addEventListener('click', () => {
                this.captureImage();
            });
        }
        
        // Configurar botón de subida si existe
        if (this.uploadButton) {
            this.uploadButton.addEventListener('click', () => {
                this.uploadImage();
            });
        }
        
        // Configurar detector corporal si existe
        if (this.bodyDetector) {
            // Listener para cuando se detecten landmarks
            this.bodyDetector.setLandmarksCallback((landmarks) => {
                // Actualizar las instrucciones según la pose actual
                this.updatePoseInstructions(landmarks);
            });
        }
        
        // Listener para cambio de unidades
        const unitToggle = document.querySelector('.unit-toggle');
        if (unitToggle) {
            unitToggle.addEventListener('change', (e) => {
                this.changeUnit(e.target.value);
            });
        }
    }
    
    /**
     * Navega al paso especificado
     * @param {number} stepIndex - Índice del paso
     */
    goToStep(stepIndex) {
        // Validar índice
        if (stepIndex < 0 || stepIndex >= this.STEPS.length) {
            return;
        }
        
        // Guardar paso actual
        this.currentStep = stepIndex;
        const step = this.STEPS[stepIndex];
        
        // Actualizar UI para este paso
        this.updateStepUI(step);
        
        // Acciones específicas según el paso
        switch (step.id) {
            case 'preparation':
                // Nada específico para este paso
                break;
                
            case 'frontpose':
                // Iniciar cámara si no está activa
                if (this.bodyDetector && !this.bodyDetector.cameraActive) {
                    this.bodyDetector.startCamera().catch(error => {
                        this.showAlert('No se pudo acceder a la cámara. ' + error.message, 'error');
                    });
                }
                break;
                
            case 'sidepose':
                // Asegurarse de que tenemos la medida frontal antes de continuar
                if (!this.measurements || !this.measurements.frontImage) {
                    this.showAlert('Necesitas capturar una imagen frontal primero', 'error');
                    this.goToStep(1); // Volver al paso frontal
                    return;
                }
                break;
                
            case 'results':
                // Detener cámara para ahorrar recursos
                if (this.bodyDetector && this.bodyDetector.cameraActive) {
                    this.bodyDetector.stopCamera();
                }
                
                // Mostrar resultados
                this.showMeasurementResults();
                
                // Mostrar botón de guardar
                this.saveButton.style.display = 'block';
                this.nextButton.style.display = 'none';
                break;
        }
        
        // Actualizar botones de navegación
        this.updateNavigationButtons();
        
        // Actualizar indicador de progreso
        this.updateProgressIndicator();
    }
    
    /**
     * Actualiza la UI para el paso actual
     * @param {Object} step - Paso actual
     */
    updateStepUI(step) {
        // Actualizar título
        this.stepTitle.textContent = step.title;
        
        // Actualizar lista de instrucciones
        this.instructionsList.innerHTML = '';
        step.instructions.forEach(instruction => {
            const li = document.createElement('li');
            li.textContent = instruction;
            this.instructionsList.appendChild(li);
        });
        
        // Actualizar imagen de referencia si existe
        if (step.poseImage) {
            this.poseImage.src = step.poseImage;
            this.poseImage.style.display = 'block';
        } else {
            this.poseImage.style.display = 'none';
        }
        
        // Ocultar alerta si hay alguna
        this.hideAlert();
    }
    
    /**
     * Actualiza los botones de navegación
     */
    updateNavigationButtons() {
        // Botón anterior
        if (this.currentStep === 0) {
            this.prevButton.style.display = 'none';
        } else {
            this.prevButton.style.display = 'block';
        }
        
        // Botón siguiente
        if (this.currentStep === this.STEPS.length - 1) {
            this.nextButton.style.display = 'none';
        } else {
            this.nextButton.style.display = 'block';
        }
        
        // Botón guardar
        if (this.currentStep === this.STEPS.length - 1) {
            this.saveButton.style.display = 'block';
        } else {
            this.saveButton.style.display = 'none';
        }
    }
    
    /**
     * Actualiza el indicador de progreso
     */
    updateProgressIndicator() {
        // Actualizar pasos
        const steps = this.progressIndicator.querySelectorAll('.progress-step');
        steps.forEach((step, index) => {
            step.classList.remove('active', 'completed');
            
            if (index === this.currentStep) {
                step.classList.add('active');
            } else if (index < this.currentStep) {
                step.classList.add('completed');
            }
        });
        
        // Actualizar línea de progreso
        const progressPercentage = (this.currentStep / (this.STEPS.length - 1)) * 100;
        this.progressLineFilled.style.width = `${progressPercentage}%`;
    }
    
    /**
     * Navega al paso siguiente
     */
    goToNextStep() {
        // Validar si podemos avanzar
        if (!this.validateCurrentStep()) {
            return;
        }
        
        // Avanzar al siguiente paso
        if (this.currentStep < this.STEPS.length - 1) {
            this.goToStep(this.currentStep + 1);
        }
    }
    
    /**
     * Navega al paso anterior
     */
    goToPreviousStep() {
        if (this.currentStep > 0) {
            this.goToStep(this.currentStep - 1);
        }
    }
    
    /**
     * Valida el paso actual
     * @returns {boolean} - Si el paso es válido
     */
    validateCurrentStep() {
        const currentStepId = this.STEPS[this.currentStep].id;
        
        switch (currentStepId) {
            case 'frontpose':
                // Verificar si tenemos una captura frontal
                if (!this.measurements || !this.measurements.frontImage) {
                    this.showAlert('Debes capturar una imagen frontal antes de continuar', 'error');
                    return false;
                }
                return true;
                
            case 'sidepose':
                // Verificar si tenemos una captura lateral
                if (!this.measurements || !this.measurements.sideImage) {
                    this.showAlert('Debes capturar una imagen lateral antes de continuar', 'error');
                    return false;
                }
                return true;
                
            default:
                return true;
        }
    }
    
    /**
     * Captura una imagen de la cámara
     */
    async captureImage() {
        if (!this.bodyDetector) {
            this.showAlert('No se ha inicializado el detector corporal', 'error');
            return;
        }
        
        if (!this.bodyDetector.cameraActive) {
            this.showAlert('La cámara no está activa', 'error');
            return;
        }
        
        try {
            // Indicar que estamos procesando
            this.measurementInProgress = true;
            this.captureButton.disabled = true;
            this.captureButton.textContent = 'Procesando...';
            
            // Capturar imagen
            const captureCanvas = await this.bodyDetector.captureImage();
            this.lastImageCapture = captureCanvas;
            
            // Procesar según el paso actual
            const currentStepId = this.STEPS[this.currentStep].id;
            
            // Inicializar measurements si no existe
            if (!this.measurements) {
                this.measurements = {
                    landmarks: {},
                    values: {}
                };
            }
            
            // Guardar imagen según la pose
            if (currentStepId === 'frontpose') {
                // Guardar imagen frontal
                this.measurements.frontImage = captureCanvas.toDataURL('image/jpeg');
                this.measurements.landmarks.front = [...this.bodyDetector.landmarks];
                
                // Calcular medidas frontales
                await this.calculateFrontMeasurements();
                
                this.showAlert('Imagen frontal capturada correctamente', 'success');
            } else if (currentStepId === 'sidepose') {
                // Guardar imagen lateral
                this.measurements.sideImage = captureCanvas.toDataURL('image/jpeg');
                this.measurements.landmarks.side = [...this.bodyDetector.landmarks];
                
                // Calcular medidas laterales
                await this.calculateSideMeasurements();
                
                this.showAlert('Imagen lateral capturada correctamente', 'success');
            }
            
            // Notificar medidas actualizadas
            this.notifyMeasurementsUpdated();
        } catch (error) {
            console.error('Error al capturar la imagen:', error);
            this.showAlert('Error al capturar la imagen: ' + error.message, 'error');
        } finally {
            // Restaurar estado
            this.measurementInProgress = false;
            this.captureButton.disabled = false;
            this.captureButton.textContent = 'Capturar';
        }
    }
    
    /**
     * Sube una imagen desde el dispositivo
     */
    uploadImage() {
        if (!this.bodyDetector) {
            this.showAlert('No se ha inicializado el detector corporal', 'error');
            return;
        }
        
        // Crear un input para seleccionar archivo
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        
        fileInput.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) {
                return;
            }
            
            try {
                // Indicar que estamos procesando
                this.measurementInProgress = true;
                this.uploadButton.disabled = true;
                this.uploadButton.textContent = 'Procesando...';
                
                // Cargar imagen
                const img = await this.fileToImage(file);
                
                // Detener cámara si está activa
                const wasCameraActive = this.bodyDetector.cameraActive;
                if (wasCameraActive) {
                    this.bodyDetector.stopCamera();
                }
                
                // Detectar landmarks en la imagen
                const results = await this.bodyDetector.detectImage(img);
                
                // Guardar última captura
                this.lastImageCapture = results.image;
                
                // Procesar según el paso actual
                const currentStepId = this.STEPS[this.currentStep].id;
                
                // Inicializar measurements si no existe
                if (!this.measurements) {
                    this.measurements = {
                        landmarks: {},
                        values: {}
                    };
                }
                
                // Guardar imagen según la pose
                if (currentStepId === 'frontpose') {
                    // Guardar imagen frontal
                    this.measurements.frontImage = results.image.toDataURL('image/jpeg');
                    this.measurements.landmarks.front = [...this.bodyDetector.landmarks];
                    
                    // Calcular medidas frontales
                    await this.calculateFrontMeasurements();
                    
                    this.showAlert('Imagen frontal subida correctamente', 'success');
                } else if (currentStepId === 'sidepose') {
                    // Guardar imagen lateral
                    this.measurements.sideImage = results.image.toDataURL('image/jpeg');
                    this.measurements.landmarks.side = [...this.bodyDetector.landmarks];
                    
                    // Calcular medidas laterales
                    await this.calculateSideMeasurements();
                    
                    this.showAlert('Imagen lateral subida correctamente', 'success');
                }
                
                // Reiniciar cámara si estaba activa
                if (wasCameraActive) {
                    await this.bodyDetector.startCamera();
                }
                
                // Notificar medidas actualizadas
                this.notifyMeasurementsUpdated();
            } catch (error) {
                console.error('Error al procesar la imagen:', error);
                this.showAlert('Error al procesar la imagen: ' + error.message, 'error');
            } finally {
                // Restaurar estado
                this.measurementInProgress = false;
                this.uploadButton.disabled = false;
                this.uploadButton.textContent = 'Subir imagen';
            }
        };
        
        // Simular clic para abrir el selector de archivos
        fileInput.click();
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
     * Calcula medidas a partir de la pose frontal
     */
    async calculateFrontMeasurements() {
        if (!this.bodyDetector || !this.measurements || !this.measurements.landmarks.front) {
            return;
        }
        
        // Obtener dimensiones del cuerpo según landmarks
        const bodyDimensions = this.calculateBodyDimensions(this.measurements.landmarks.front);
        
        // Inicializar objeto de valores si no existe
        if (!this.measurements.values) {
            this.measurements.values = {};
        }
        
        // Guardar medidas
        this.measurements.values = {
            ...this.measurements.values,
            ...bodyDimensions
        };
        
        // Mostrar resultados preliminares
        this.showMeasurementResults();
    }
    
    /**
     * Calcula medidas a partir de la pose lateral
     */
    async calculateSideMeasurements() {
        if (!this.bodyDetector || !this.measurements || !this.measurements.landmarks.side) {
            return;
        }
        
        // Obtener dimensiones laterales
        const sideDimensions = this.calculateSideBodyDimensions(this.measurements.landmarks.side);
        
        // Actualizar medidas existentes
        this.measurements.values = {
            ...this.measurements.values,
            ...sideDimensions
        };
        
        // Calcular medidas combinadas (usando datos frontales y laterales)
        if (this.measurements.landmarks.front) {
            this.calculateCombinedMeasurements();
        }
        
        // Mostrar resultados actualizados
        this.showMeasurementResults();
    }
    
    /**
     * Calcula dimensiones corporales a partir de landmarks frontales
     * @param {Array} landmarks - Array de landmarks
     * @returns {Object} - Dimensiones calculadas
     */
    calculateBodyDimensions(landmarks) {
        // Esta función debería implementar algoritmos para convertir
        // distancias entre landmarks a medidas reales en cm o pulgadas
        
        // Ejemplo simplificado:
        const dimensions = {};
        
        // Factor de conversión (esto debería calibrarse según la estatura real del usuario)
        // Como aproximación, podemos usar la altura estándar
        const userHeight = 170; // cm
        const pixelHeight = this.bodyDetector.getDistanceInPixels(0, 27); // Distancia nariz-tobillo
        const conversionFactor = userHeight / pixelHeight;
        
        // Medidas de hombros
        const shoulderWidth = this.bodyDetector.getDistanceInPixels(11, 12);
        dimensions.shoulders = Math.round(shoulderWidth * conversionFactor * 10) / 10;
        
        // Medidas de pecho
        // Aproximación: distancia entre landmarks de hombros * factor
        const chestWidth = shoulderWidth * 0.9;
        dimensions.chest = Math.round(chestWidth * conversionFactor * 10) / 10;
        
        // Medidas de cintura
        const waistWidth = this.bodyDetector.getDistanceInPixels(23, 24);
        dimensions.waist = Math.round(waistWidth * conversionFactor * 10) / 10;
        
        // Medidas de cadera
        // Aproximación: distancia entre caderas * factor
        const hipWidth = waistWidth * 1.2;
        dimensions.hips = Math.round(hipWidth * conversionFactor * 10) / 10;
        
        // Longitud de brazos
        const armLength = 
            this.bodyDetector.getDistanceInPixels(11, 13) + 
            this.bodyDetector.getDistanceInPixels(13, 15);
        dimensions.armLength = Math.round(armLength * conversionFactor * 10) / 10;
        
        // Altura de entrepierna
        const inseamHeight = this.bodyDetector.getDistanceInPixels(
            Math.floor((23 + 24) / 2), // Punto medio entre caderas
            27 // Tobillo
        );
        dimensions.inseam = Math.round(inseamHeight * conversionFactor * 10) / 10;
        
        return dimensions;
    }
    
    /**
     * Calcula dimensiones corporales a partir de landmarks laterales
     * @param {Array} landmarks - Array de landmarks
     * @returns {Object} - Dimensiones calculadas
     */
    calculateSideBodyDimensions(landmarks) {
        // Cálculos basados en la vista lateral
        const dimensions = {};
        
        // Factor de conversión (usando el mismo que en la vista frontal)
        const userHeight = 170; // cm
        const pixelHeight = this.bodyDetector.getDistanceInPixels(0, 27); // Distancia nariz-tobillo
        const conversionFactor = userHeight / pixelHeight;
        
        // Profundidad del pecho (vista lateral)
        // Aproximación basada en la distancia entre hombro y pecho en vista lateral
        const chestDepth = this.bodyDetector.getDistanceInPixels(11, 12) * 0.3;
        dimensions.chestDepth = Math.round(chestDepth * conversionFactor * 10) / 10;
        
        // Profundidad de la cintura
        const waistDepth = this.bodyDetector.getDistanceInPixels(23, 24) * 0.25;
        dimensions.waistDepth = Math.round(waistDepth * conversionFactor * 10) / 10;
        
        return dimensions;
    }
    
    /**
     * Calcula medidas combinadas usando datos frontales y laterales
     */
    calculateCombinedMeasurements() {
        if (!this.measurements || !this.measurements.values) {
            return;
        }
        
        const values = this.measurements.values;
        
        // Circunferencia de pecho (aproximación)
        if (values.chest && values.chestDepth) {
            // Fórmula simplificada: 2 * (ancho + profundidad)
            values.chestCircumference = Math.round((2 * (values.chest + values.chestDepth)) * 10) / 10;
        }
        
        // Circunferencia de cintura (aproximación)
        if (values.waist && values.waistDepth) {
            // Fórmula simplificada: 2 * (ancho + profundidad)
            values.waistCircumference = Math.round((2 * (values.waist + values.waistDepth)) * 10) / 10;
        }
        
        // Tallas estimadas
        values.shirtSize = this.estimateShirtSize(values);
        values.pantsSize = this.estimatePantsSize(values);
    }
    
    /**
     * Estima la talla de camisa
     * @param {Object} measurements - Medidas calculadas
     * @returns {string} - Talla estimada
     */
    estimateShirtSize(measurements) {
        // Ejemplo básico basado en circunferencia de pecho
        if (!measurements.chestCircumference) {
            return 'Desconocido';
        }
        
        const chest = measurements.chestCircumference;
        
        // Tallas para hombres (aproximación)
        if (chest < 89) return 'XS';
        if (chest < 97) return 'S';
        if (chest < 105) return 'M';
        if (chest < 113) return 'L';
        if (chest < 121) return 'XL';
        return 'XXL';
    }
    
    /**
     * Estima la talla de pantalón
     * @param {Object} measurements - Medidas calculadas
     * @returns {string} - Talla estimada
     */
    estimatePantsSize(measurements) {
        // Ejemplo básico basado en circunferencia de cintura
        if (!measurements.waistCircumference) {
            return 'Desconocido';
        }
        
        const waist = measurements.waistCircumference;
        
        // Convertir a pulgadas para tallas estándar
        const waistInches = Math.round(waist / 2.54);
        
        // Redondear a la talla más cercana (típicamente en incrementos de 2)
        return Math.floor(waistInches / 2) * 2;
    }
    
    /**
     * Muestra los resultados de las mediciones
     */
    showMeasurementResults() {
        if (!this.measurementGrid || !this.measurements || !this.measurements.values) {
            return;
        }
        
        // Limpiar grid
        this.measurementGrid.innerHTML = '';
        
        // Etiquetas amigables para las medidas
        const labels = {
            shoulders: 'Hombros',
            chest: 'Ancho de pecho',
            chestCircumference: 'Contorno de pecho',
            waist: 'Ancho de cintura',
            waistCircumference: 'Contorno de cintura',
            hips: 'Cadera',
            armLength: 'Longitud de brazo',
            inseam: 'Entrepierna',
            shirtSize: 'Talla de camisa estimada',
            pantsSize: 'Talla de pantalón estimada'
        };
        
        // Unidad de medida
        const unit = this.UNIT;
        
        // Crear elementos para cada medida
        Object.entries(this.measurements.values).forEach(([key, value]) => {
            // Omitir propiedades que no son medidas
            if (!labels[key]) {
                return;
            }
            
            const itemElement = document.createElement('div');
            itemElement.className = 'measurement-item';
            itemElement.setAttribute('data-measurement', key);
            
            const labelElement = document.createElement('div');
            labelElement.className = 'measurement-label';
            labelElement.textContent = labels[key] || key;
            
            const valueElement = document.createElement('div');
            valueElement.className = 'measurement-value';
            
            // Formatear valor según el tipo de medida
            if (key === 'shirtSize' || key === 'pantsSize') {
                valueElement.textContent = value;
            } else {
                valueElement.textContent = value;
                
                const unitElement = document.createElement('span');
                unitElement.className = 'measurement-unit';
                unitElement.textContent = unit;
                valueElement.appendChild(unitElement);
            }
            
            // Añadir posibilidad de editar el valor
            valueElement.addEventListener('click', () => {
                this.editMeasurementValue(key, value, itemElement);
            });
            
            itemElement.appendChild(labelElement);
            itemElement.appendChild(valueElement);
            
            this.measurementGrid.appendChild(itemElement);
        });
    }
    
    /**
     * Permite editar un valor de medición
     * @param {string} key - Clave de la medida
     * @param {number|string} currentValue - Valor actual
     * @param {HTMLElement} container - Contenedor del elemento
     */
    editMeasurementValue(key, currentValue, container) {
        // Verificar si ya está en modo edición
        if (container.querySelector('input')) {
            return;
        }
        
        // Obtener el elemento de valor
        const valueElement = container.querySelector('.measurement-value');
        
        // Guardar el contenido HTML original
        const originalContent = valueElement.innerHTML;
        
        // Crear input para edición
        const input = document.createElement('input');
        input.type = 'text';
        input.value = currentValue;
        input.className = 'measurement-input';
        
        // Reemplazar contenido
        valueElement.innerHTML = '';
        valueElement.appendChild(input);
        
        // Enfocar input
        input.focus();
        input.select();
        
        // Manejar confirmación (Enter o perder foco)
        const confirmEdit = () => {
            const newValue = input.value.trim();
            
            // Validar que sea un número para medidas numéricas
            if (['shirtSize', 'pantsSize'].includes(key)) {
                // Para tallas, aceptar valores alfanuméricos
                this.measurements.values[key] = newValue;
            } else {
                // Para medidas, convertir a número
                const numValue = parseFloat(newValue);
                if (!isNaN(numValue)) {
                    this.measurements.values[key] = Math.round(numValue * 10) / 10;
                }
            }
            
            // Actualizar la UI
            this.showMeasurementResults();
            
            // Notificar actualización
            this.notifyMeasurementsUpdated();
        };
        
        // Manejar cancelación (Escape)
        const cancelEdit = () => {
            valueElement.innerHTML = originalContent;
        };
        
        // Event listeners
        input.addEventListener('blur', confirmEdit);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                confirmEdit();
            } else if (e.key === 'Escape') {
                cancelEdit();
            }
        });
    }
    
    /**
     * Cambia la unidad de medida
     * @param {string} newUnit - Nueva unidad (cm o in)
     */
    changeUnit(newUnit) {
        if (newUnit !== 'cm' && newUnit !== 'in') {
            return;
        }
        
        const oldUnit = this.UNIT;
        this.UNIT = newUnit;
        
        // Si tenemos medidas, convertirlas
        if (this.measurements && this.measurements.values) {
            const values = this.measurements.values;
            
            // Convertir cada valor numérico
            Object.keys(values).forEach(key => {
                // Omitir valores no numéricos
                if (typeof values[key] !== 'number') {
                    return;
                }
                
                if (oldUnit === 'cm' && newUnit === 'in') {
                    // Convertir de cm a pulgadas
                    values[key] = Math.round((values[key] / 2.54) * 10) / 10;
                } else if (oldUnit === 'in' && newUnit === 'cm') {
                    // Convertir de pulgadas a cm
                    values[key] = Math.round((values[key] * 2.54) * 10) / 10;
                }
            });
            
            // Actualizar la UI
            this.showMeasurementResults();
        }
    }
    
    /**
     * Actualiza instrucciones según la pose actual
     * @param {Array} landmarks - Landmarks detectados
     */
    updatePoseInstructions(landmarks) {
        if (!landmarks || this.measurementInProgress) {
            return;
        }
        
        const currentStepId = this.STEPS[this.currentStep].id;
        
        // Solo dar feedback en pasos de captura
        if (currentStepId !== 'frontpose' && currentStepId !== 'sidepose') {
            return;
        }
        
        // Verificar posición y dar retroalimentación
        const feedback = this.getPoseFeedback(landmarks, currentStepId);
        
        // Mostrar retroalimentación si es necesario
        if (feedback) {
            this.showAlert(feedback, 'info');
        } else {
            this.hideAlert();
        }
    }
    
    /**
     * Obtiene retroalimentación sobre la pose actual
     * @param {Array} landmarks - Landmarks detectados
     * @param {string} poseType - Tipo de pose esperada
     * @returns {string|null} - Mensaje de retroalimentación o null
     */
    getPoseFeedback(landmarks, poseType) {
        // Implementación simplificada para verificar si la pose es correcta
        
        // Verificar si hay suficientes landmarks
        if (!landmarks || landmarks.length < 33) {
            return "No se detecta el cuerpo completo. Asegúrate de estar completamente visible.";
        }
        
        // Verificar visibilidad de landmarks clave
        const keyLandmarks = [0, 11, 12, 23, 24, 27, 28]; // Nariz, hombros, caderas, tobillos
        const lowVisibility = keyLandmarks.some(index => landmarks[index].visibility < 0.5);
        
        if (lowVisibility) {
            return "Algunas partes importantes de tu cuerpo no son claramente visibles.";
        }
        
        if (poseType === 'frontpose') {
            // Verificar si está de frente
            // Comprobar que los hombros están aproximadamente a la misma altura
            const shoulderYDiff = Math.abs(landmarks[11].y - landmarks[12].y);
            if (shoulderYDiff > 0.05) {
                return "Intenta estar más derecho, con los hombros nivelados.";
            }
            
            // Comprobar que está mirando a la cámara (hombros aproximadamente a la misma distancia de la nariz)
            const leftShoulderX = landmarks[11].x;
            const rightShoulderX = landmarks[12].x;
            const noseX = landmarks[0].x;
            
            const leftDist = Math.abs(noseX - leftShoulderX);
            const rightDist = Math.abs(noseX - rightShoulderX);
            const shoulderXRatio = Math.max(leftDist, rightDist) / Math.min(leftDist, rightDist);
            
            if (shoulderXRatio > 1.3) {
                return "Intenta mirar directamente a la cámara, con tu cuerpo orientado al frente.";
            }
        } else if (poseType === 'sidepose') {
            // Verificar si está de lado
            // En posición lateral, un hombro debería estar claramente delante del otro
            const leftShoulderX = landmarks[11].x;
            const rightShoulderX = landmarks[12].x;
            
            const shoulderXDiff = Math.abs(leftShoulderX - rightShoulderX);
            if (shoulderXDiff < 0.1) {
                return "Gira más tu cuerpo para obtener una vista lateral adecuada.";
            }
        }
        
        // Si llega aquí, la pose parece correcta
        return null;
    }
    
    /**
     * Muestra un mensaje de alerta
     * @param {string} message - Mensaje a mostrar
     * @param {string} type - Tipo de alerta (error, info, success)
     */
    showAlert(message, type = 'info') {
        if (!this.alertMessage) {
            return;
        }
        
        // Configurar clase según tipo
        this.alertMessage.className = `alert-message ${type}-message`;
        
        // Añadir icono según tipo
        let icon = '';
        switch (type) {
            case 'error':
                icon = '<i class="fas fa-exclamation-circle"></i>';
                break;
            case 'success':
                icon = '<i class="fas fa-check-circle"></i>';
                break;
            case 'info':
                icon = '<i class="fas fa-info-circle"></i>';
                break;
        }
        
        // Establecer contenido
        this.alertMessage.innerHTML = `${icon} ${message}`;
        
        // Mostrar
        this.alertMessage.style.display = 'flex';
        
        // Auto-ocultar después de un tiempo para alertas success
        if (type === 'success') {
            setTimeout(() => {
                this.hideAlert();
            }, 5000);
        }
    }
    
    /**
     * Oculta el mensaje de alerta
     */
    hideAlert() {
        if (this.alertMessage) {
            this.alertMessage.style.display = 'none';
        }
    }
    
    /**
     * Guarda las mediciones y notifica
     */
    saveMeasurements() {
        if (!this.measurements || !this.measurements.values) {
            this.showAlert('No hay medidas para guardar', 'error');
            return;
        }
        
        try {
            // Guardar localmente
            localStorage.setItem('userMeasurements', JSON.stringify(this.measurements));
            
            // Notificar que las medidas están completas
            if (this.onMeasurementsComplete) {
                this.onMeasurementsComplete(this.measurements);
            }
            
            this.showAlert('Medidas guardadas correctamente', 'success');
            
            // Redirigir a la página de prueba virtual después de unos segundos
            setTimeout(() => {
                window.location.href = '/fitting';
            }, 2000);
        } catch (error) {
            console.error('Error al guardar las medidas:', error);
            this.showAlert('Error al guardar las medidas: ' + error.message, 'error');
        }
    }
    
    /**
     * Notifica que las medidas han sido actualizadas
     */
    notifyMeasurementsUpdated() {
        if (this.onMeasurementsUpdated) {
            this.onMeasurementsUpdated(this.measurements);
        }
    }
    
    /**
     * Configura el callback para cuando se completan las medidas
     * @param {Function} callback - Función de callback
     */
    onComplete(callback) {
        if (typeof callback === 'function') {
            this.onMeasurementsComplete = callback;
        }
    }
    
    /**
     * Configura el callback para cuando se actualizan las medidas
     * @param {Function} callback - Función de callback
     */
    onUpdate(callback) {
        if (typeof callback === 'function') {
            this.onMeasurementsUpdated = callback;
        }
    }
    
    /**
     * Carga medidas previas
     * @returns {Object|null} - Medidas cargadas o null
     */
    loadPreviousMeasurements() {
        try {
            const savedMeasurements = localStorage.getItem('userMeasurements');
            if (savedMeasurements) {
                this.measurements = JSON.parse(savedMeasurements);
                this.showMeasurementResults();
                return this.measurements;
            }
        } catch (error) {
            console.error('Error al cargar medidas previas:', error);
        }
        
        return null;
    }
    
    /**
     * Limpia el estado actual
     */
    reset() {
        this.measurements = null;
        this.lastImageCapture = null;
        this.currentStep = 0;
        
        // Actualizar UI
        this.goToStep(0);
        this.measurementGrid.innerHTML = '';
    }
}

// Exportar la clase para uso en otros módulos
export default MeasurementUI;