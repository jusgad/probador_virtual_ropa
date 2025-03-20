/**
 * clothingFitter.js - Módulo para ajustar prendas de vestir al cuerpo
 * 
 * Este módulo se encarga de:
 * - Cargar imágenes de prendas de vestir
 * - Transformar y ajustar prendas a los landmarks corporales
 * - Renderizar la visualización de prendas ajustadas al cuerpo
 * - Gestionar tipos de prendas (camisas, pantalones, etc.)
 */

/**
 * Clase principal para el ajuste de prendas
 */
class ClothingFitter {
    /**
     * Constructor
     * @param {Object} options - Opciones de configuración
     * @param {HTMLElement} options.container - Contenedor para la visualización
     * @param {Object} options.measurements - Medidas del usuario
     */
    constructor(options = {}) {
        // Elementos del DOM
        this.container = options.container || null;
        this.canvas = null;
        this.ctx = null;
        
        // Dimensiones
        this.width = options.width || 640;
        this.height = options.height || 480;
        
        // Medidas del usuario
        this.measurements = options.measurements || null;
        
        // Estado
        this.isInitialized = false;
        this.loading = false;
        
        // Prendas actuales
        this.currentClothing = null;
        this.clothingLayers = {}; // Múltiples prendas por capa (torso, piernas, etc.)
        
        // Referencia de cuerpo
        this.bodyImage = null;
        this.bodyLandmarks = null;
        
        // Referencia para manipulación de imágenes
        this.tempCanvas = document.createElement('canvas');
        this.tempCtx = this.tempCanvas.getContext('2d');
        
        // Callbacks
        this.onFittingComplete = null;
        this.onFittingUpdated = null;
        
        // Opciones de renderizado
        this.options = {
            showBodyOutline: true,
            showLandmarks: false,
            showSizeIndicators: true,
            showLabels: true,
            fitQuality: 'high',  // low, medium, high
            renderMode: '2d'     // 2d o 3d (si hay soporte WebGL)
        };
        
        // Tipos de prendas soportadas
        this.clothingTypes = {
            SHIRT: 'shirt',
            TSHIRT: 't-shirt',
            PANTS: 'pants',
            SHORTS: 'shorts',
            SKIRT: 'skirt',
            DRESS: 'dress',
            JACKET: 'jacket',
            SWEATER: 'sweater',
            HAT: 'hat',
            SHOES: 'shoes'
        };
        
        // Inicialización automática si hay un contenedor
        if (this.container) {
            this.initialize();
        }
    }
    
    /**
     * Inicializa el ajustador de prendas
     * @returns {Promise<boolean>} - Éxito de la inicialización
     */
    async initialize() {
        if (this.isInitialized) {
            return true;
        }
        
        try {
            console.log("Inicializando ajustador de prendas...");
            
            // Crear canvas si no existe
            if (!this.canvas) {
                this.canvas = document.createElement('canvas');
                this.canvas.width = this.width;
                this.canvas.height = this.height;
                this.canvas.className = 'fitting-canvas';
                this.ctx = this.canvas.getContext('2d');
                
                // Añadir al contenedor
                if (this.container) {
                    this.container.appendChild(this.canvas);
                }
            }
            
            // Crear estructura de UI adicional
            this.createUI();
            
            // Cargar imagen de maniquí por defecto
            await this.loadDefaultModel();
            
            this.isInitialized = true;
            console.log("Ajustador de prendas inicializado correctamente");
            
            // Dibujar estado inicial
            this.render();
            
            return true;
        } catch (error) {
            console.error("Error al inicializar el ajustador de prendas:", error);
            return false;
        }
    }
    
    /**
     * Crea elementos UI adicionales
     */
    createUI() {
        // Contenedor de controles
        this.controlsContainer = document.createElement('div');
        this.controlsContainer.className = 'fitting-controls';
        
        // Selector de vista
        this.viewSelector = document.createElement('div');
        this.viewSelector.className = 'view-selector';
        this.viewSelector.innerHTML = `
            <button class="view-button active" data-view="front">Frente</button>
            <button class="view-button" data-view="side">Lado</button>
            <button class="view-button" data-view="back">Espalda</button>
        `;
        
        // Añadir event listeners para selector de vista
        this.viewSelector.querySelectorAll('.view-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const view = e.target.getAttribute('data-view');
                this.changeView(view);
                
                // Actualizar botón activo
                this.viewSelector.querySelectorAll('.view-button').forEach(btn => 
                    btn.classList.remove('active'));
                e.target.classList.add('active');
            });
        });
        
        // Etiqueta de carga
        this.loadingLabel = document.createElement('div');
        this.loadingLabel.className = 'loading-label';
        this.loadingLabel.innerHTML = 'Cargando...';
        this.loadingLabel.style.display = 'none';
        
        // Etiqueta de talla
        this.sizeLabel = document.createElement('div');
        this.sizeLabel.className = 'size-label';
        this.sizeLabel.innerHTML = '';
        
        // Añadir elementos al contenedor
        this.controlsContainer.appendChild(this.viewSelector);
        this.container.appendChild(this.controlsContainer);
        this.container.appendChild(this.loadingLabel);
        this.container.appendChild(this.sizeLabel);
    }
    
    /**
     * Carga un modelo de cuerpo por defecto
     * @returns {Promise<boolean>} - Éxito de la carga
     */
    async loadDefaultModel() {
        try {
            // Cargar imagen de maniquí
            const bodyModelImage = new Image();
            bodyModelImage.src = '/static/img/placeholder_user.png';
            
            return new Promise((resolve, reject) => {
                bodyModelImage.onload = () => {
                    this.bodyImage = bodyModelImage;
                    
                    // Definir landmarks de referencia para el maniquí
                    this.bodyLandmarks = this.getDefaultBodyLandmarks();
                    
                    resolve(true);
                };
                
                bodyModelImage.onerror = () => {
                    console.warn("No se pudo cargar el modelo por defecto, usando silueta básica");
                    // Usar silueta básica en su lugar
                    this.bodyImage = null;
                    this.bodyLandmarks = this.getDefaultBodyLandmarks();
                    resolve(false);
                };
            });
        } catch (error) {
            console.error("Error al cargar el modelo por defecto:", error);
            return false;
        }
    }
    
    /**
     * Obtiene landmarks de cuerpo por defecto
     * @returns {Array} - Array de landmarks por defecto
     */
    getDefaultBodyLandmarks() {
        // Landmarks básicos para un cuerpo de referencia
        // Estas proporciones están basadas en un modelo humano promedio
        // Las coordenadas están normalizadas (0-1)
        
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        return [
            // Cara
            { x: 0.5, y: 0.05, z: 0, visibility: 1.0 }, // 0: nariz
            
            // Hombros
            { x: 0.35, y: 0.2, z: 0, visibility: 1.0 }, // 1: hombro izquierdo
            { x: 0.65, y: 0.2, z: 0, visibility: 1.0 }, // 2: hombro derecho
            
            // Codos
            { x: 0.25, y: 0.35, z: 0, visibility: 1.0 }, // 3: codo izquierdo
            { x: 0.75, y: 0.35, z: 0, visibility: 1.0 }, // 4: codo derecho
            
            // Muñecas
            { x: 0.2, y: 0.5, z: 0, visibility: 1.0 }, // 5: muñeca izquierda
            { x: 0.8, y: 0.5, z: 0, visibility: 1.0 }, // 6: muñeca derecha
            
            // Caderas
            { x: 0.4, y: 0.5, z: 0, visibility: 1.0 }, // 7: cadera izquierda
            { x: 0.6, y: 0.5, z: 0, visibility: 1.0 }, // 8: cadera derecha
            
            // Rodillas
            { x: 0.4, y: 0.7, z: 0, visibility: 1.0 }, // 9: rodilla izquierda
            { x: 0.6, y: 0.7, z: 0, visibility: 1.0 }, // 10: rodilla derecha
            
            // Tobillos
            { x: 0.4, y: 0.9, z: 0, visibility: 1.0 }, // 11: tobillo izquierdo
            { x: 0.6, y: 0.9, z: 0, visibility: 1.0 }  // 12: tobillo derecho
        ];
    }
    
    /**
     * Cambia la vista actual (frente, lado, espalda)
     * @param {string} view - Vista a mostrar
     */
    changeView(view) {
        // Actualizar vista actual
        this.currentView = view;
        
        // Re-renderizar con la nueva vista
        this.render();
    }
    
    /**
     * Configura nuevas medidas de usuario
     * @param {Object} measurements - Nuevas medidas
     */
    setMeasurements(measurements) {
        this.measurements = measurements;
        
        // Actualizar landmarks si es necesario
        if (measurements.landmarks) {
            this.bodyLandmarks = measurements.landmarks;
        }
        
        // Re-ajustar las prendas actuales con las nuevas medidas
        if (this.currentClothing) {
            this.fitClothing(this.currentClothing);
        } else {
            this.render();
        }
    }
    
    /**
     * Configura una imagen de cuerpo del usuario
     * @param {HTMLImageElement|HTMLCanvasElement} bodyImage - Imagen del cuerpo
     * @param {Array} landmarks - Landmarks detectados
     */
    setBodyImage(bodyImage, landmarks) {
        this.bodyImage = bodyImage;
        
        if (landmarks) {
            this.bodyLandmarks = landmarks;
        }
        
        // Re-renderizar con la nueva imagen de cuerpo
        this.render();
    }
    
    /**
     * Carga y ajusta una prenda al cuerpo
     * @param {Object} clothing - Información de la prenda
     * @returns {Promise<boolean>} - Éxito del ajuste
     */
    async fitClothing(clothing) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        this.showLoading(true);
        
        try {
            console.log("Ajustando prenda:", clothing.name);
            
            // Guardar referencia a la prenda actual
            this.currentClothing = clothing;
            
            // Cargar imagen de la prenda
            const clothingImage = await this.loadClothingImage(clothing);
            
            // Determinar tipo de prenda y capa
            const clothingType = this.determineClothingType(clothing);
            const layer = this.getLayerForClothingType(clothingType);
            
            // Crear objeto de prenda ajustada
            const fittedClothing = {
                image: clothingImage,
                type: clothingType,
                info: clothing,
                transformation: {
                    scale: this.calculateScaleFactor(clothing),
                    position: this.getClothingPosition(clothingType),
                    rotation: 0,
                    opacity: 1
                },
                fit: this.calculateFitData(clothing)
            };
            
            // Añadir a la capa correspondiente
            this.clothingLayers[layer] = fittedClothing;
            
            // Renderizar
            this.render();
            
            // Actualizar etiqueta de talla
            this.updateSizeLabel(clothing);
            
            // Notificar que se completó el ajuste
            if (this.onFittingComplete) {
                this.onFittingComplete(fittedClothing);
            }
            
            this.showLoading(false);
            return true;
        } catch (error) {
            console.error("Error al ajustar la prenda:", error);
            this.showLoading(false);
            return false;
        }
    }
    
    /**
     * Carga la imagen de una prenda
     * @param {Object} clothing - Información de la prenda
     * @returns {Promise<HTMLImageElement>} - Imagen cargada
     */
    loadClothingImage(clothing) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            
            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error(`No se pudo cargar la imagen: ${clothing.image}`));
            
            // Determinar la URL de la imagen
            let imageUrl = clothing.image;
            
            // Si hay una imagen específica para la talla, usarla
            if (clothing.sizes && clothing.selectedSize && 
                clothing.sizes[clothing.selectedSize] && 
                clothing.sizes[clothing.selectedSize].image) {
                imageUrl = clothing.sizes[clothing.selectedSize].image;
            }
            
            image.src = imageUrl;
        });
    }
    
    /**
     * Determina el tipo de prenda
     * @param {Object} clothing - Información de la prenda
     * @returns {string} - Tipo de prenda
     */
    determineClothingType(clothing) {
        // Si la prenda ya tiene un tipo definido, usarlo
        if (clothing.type && this.clothingTypes[clothing.type.toUpperCase()]) {
            return this.clothingTypes[clothing.type.toUpperCase()];
        }
        
        // Intentar determinar por nombre/categoría
        const name = clothing.name.toLowerCase();
        const category = clothing.category ? clothing.category.toLowerCase() : '';
        
        if (name.includes('camisa') || category.includes('camisa')) {
            return this.clothingTypes.SHIRT;
        } else if (name.includes('camiseta') || name.includes('t-shirt') || 
                  category.includes('camiseta')) {
            return this.clothingTypes.TSHIRT;
        } else if (name.includes('pantalón') || category.includes('pantalón')) {
            return this.clothingTypes.PANTS;
        } else if (name.includes('short') || name.includes('bermuda') || 
                  category.includes('short')) {
            return this.clothingTypes.SHORTS;
        } else if (name.includes('falda') || category.includes('falda')) {
            return this.clothingTypes.SKIRT;
        } else if (name.includes('vestido') || category.includes('vestido')) {
            return this.clothingTypes.DRESS;
        } else if (name.includes('chaqueta') || name.includes('jacket') || 
                  category.includes('chaqueta')) {
            return this.clothingTypes.JACKET;
        } else if (name.includes('suéter') || name.includes('sweater') || 
                  category.includes('suéter')) {
            return this.clothingTypes.SWEATER;
        }
        
        // Si no se puede determinar, asumir camiseta (más común)
        return this.clothingTypes.TSHIRT;
    }
    
    /**
     * Obtiene la capa para un tipo de prenda
     * @param {string} type - Tipo de prenda
     * @returns {string} - Capa de renderizado
     */
    getLayerForClothingType(type) {
        // Definir capas para controlar el orden de renderizado
        switch (type) {
            case this.clothingTypes.PANTS:
            case this.clothingTypes.SHORTS:
            case this.clothingTypes.SKIRT:
                return 'bottom';
                
            case this.clothingTypes.SHIRT:
            case this.clothingTypes.TSHIRT:
                return 'middle';
                
            case this.clothingTypes.JACKET:
            case this.clothingTypes.SWEATER:
                return 'top';
                
            case this.clothingTypes.DRESS:
                return 'full';
                
            case this.clothingTypes.HAT:
                return 'head';
                
            case this.clothingTypes.SHOES:
                return 'feet';
                
            default:
                return 'middle';
        }
    }
    
    /**
     * Calcula el factor de escala para la prenda
     * @param {Object} clothing - Información de la prenda
     * @returns {number} - Factor de escala
     */
    calculateScaleFactor(clothing) {
        // Si no hay medidas, usar escala por defecto
        if (!this.measurements) {
            return 1.0;
        }
        
        // Tipo de prenda
        const type = this.determineClothingType(clothing);
        
        // Talla seleccionada
        const selectedSize = clothing.selectedSize || 'M';
        
        // Obtener dimensiones de referencia según tipo de prenda
        let userMeasurement = 0;
        let clothingMeasurement = 0;
        
        switch (type) {
            case this.clothingTypes.SHIRT:
            case this.clothingTypes.TSHIRT:
            case this.clothingTypes.JACKET:
            case this.clothingTypes.SWEATER:
                // Para prendas superiores, usar medida de pecho
                userMeasurement = this.measurements.chest || 100;
                clothingMeasurement = this.getClothingSizeMeasurement(clothing, selectedSize, 'chest') || 100;
                break;
                
            case this.clothingTypes.PANTS:
            case this.clothingTypes.SHORTS:
            case this.clothingTypes.SKIRT:
                // Para prendas inferiores, usar medida de cintura
                userMeasurement = this.measurements.waist || 80;
                clothingMeasurement = this.getClothingSizeMeasurement(clothing, selectedSize, 'waist') || 80;
                break;
                
            case this.clothingTypes.DRESS:
                // Para vestidos, combinar pecho y cintura
                const chestMeasurement = this.measurements.chest || 100;
                const waistMeasurement = this.measurements.waist || 80;
                userMeasurement = (chestMeasurement + waistMeasurement) / 2;
                
                const chestClothing = this.getClothingSizeMeasurement(clothing, selectedSize, 'chest') || 100;
                const waistClothing = this.getClothingSizeMeasurement(clothing, selectedSize, 'waist') || 80;
                clothingMeasurement = (chestClothing + waistClothing) / 2;
                break;
                
            default:
                // Escala por defecto para otros tipos
                return 1.0;
        }
        
        // Calcular factor de escala
        // Una relación de 1:1 entre medidas daría una escala de 1.0
        const scaleFactor = userMeasurement / clothingMeasurement;
        
        // Limitar el factor de escala para evitar deformaciones extremas
        return Math.max(0.8, Math.min(1.2, scaleFactor));
    }
    
    /**
     * Obtiene la medida de una prenda para una talla específica
     * @param {Object} clothing - Información de la prenda
     * @param {string} size - Talla
     * @param {string} measurementType - Tipo de medida
     * @returns {number} - Medida en cm
     */
    getClothingSizeMeasurement(clothing, size, measurementType) {
        // Verificar si la prenda tiene información de tallas
        if (clothing.sizes && clothing.sizes[size] && 
            clothing.sizes[size].measurements && 
            clothing.sizes[size].measurements[measurementType]) {
            return clothing.sizes[size].measurements[measurementType];
        }
        
        // Si no hay datos específicos, usar valores por defecto según tipo y talla
        const defaultMeasurements = this.getDefaultSizeMeasurements();
        const type = this.determineClothingType(clothing);
        
        if (defaultMeasurements[type] && 
            defaultMeasurements[type][size] && 
            defaultMeasurements[type][size][measurementType]) {
            return defaultMeasurements[type][size][measurementType];
        }
        
        // Valores por defecto si todo lo demás falla
        const defaultValues = {
            'chest': 100,
            'waist': 80,
            'hips': 100,
            'shoulder': 45,
            'sleeve': 65,
            'length': 70,
            'inseam': 80
        };
        
        return defaultValues[measurementType] || 100;
    }
    
    /**
     * Obtiene medidas por defecto para diferentes tipos de prendas y tallas
     * @returns {Object} - Tabla de medidas por defecto
     */
    getDefaultSizeMeasurements() {
        return {
            'shirt': {
                'XS': { 'chest': 90, 'waist': 80, 'shoulder': 42, 'sleeve': 63, 'length': 68 },
                'S': { 'chest': 94, 'waist': 84, 'shoulder': 44, 'sleeve': 64, 'length': 70 },
                'M': { 'chest': 100, 'waist': 90, 'shoulder': 46, 'sleeve': 65, 'length': 72 },
                'L': { 'chest': 106, 'waist': 96, 'shoulder': 48, 'sleeve': 66, 'length': 74 },
                'XL': { 'chest': 112, 'waist': 102, 'shoulder': 50, 'sleeve': 67, 'length': 76 }
            },
            't-shirt': {
                'XS': { 'chest': 88, 'shoulder': 41, 'length': 66 },
                'S': { 'chest': 92, 'shoulder': 43, 'length': 68 },
                'M': { 'chest': 98, 'shoulder': 45, 'length': 70 },
                'L': { 'chest': 104, 'shoulder': 47, 'length': 72 },
                'XL': { 'chest': 110, 'shoulder': 49, 'length': 74 }
            },
            'pants': {
                '28': { 'waist': 72, 'hips': 90, 'inseam': 78 },
                '30': { 'waist': 76, 'hips': 94, 'inseam': 79 },
                '32': { 'waist': 81, 'hips': 98, 'inseam': 80 },
                '34': { 'waist': 86, 'hips': 102, 'inseam': 81 },
                '36': { 'waist': 91, 'hips': 106, 'inseam': 82 }
            },
            'dress': {
                'XS': { 'chest': 84, 'waist': 66, 'hips': 91, 'length': 90 },
                'S': { 'chest': 88, 'waist': 70, 'hips': 95, 'length': 92 },
                'M': { 'chest': 94, 'waist': 76, 'hips': 101, 'length': 94 },
                'L': { 'chest': 100, 'waist': 83, 'hips': 107, 'length': 96 },
                'XL': { 'chest': 106, 'waist': 90, 'hips': 114, 'length': 98 }
            }
        };
    }
    
    /**
     * Obtiene la posición para un tipo de prenda
     * @param {string} type - Tipo de prenda
     * @returns {Object} - Coordenadas x,y
     */
    getClothingPosition(type) {
        // Posiciones relativas al canvas para diferentes tipos de prendas
        switch (type) {
            case this.clothingTypes.SHIRT:
            case this.clothingTypes.TSHIRT:
            case this.clothingTypes.JACKET:
            case this.clothingTypes.SWEATER:
                // Prendas superiores: centradas en el pecho
                return { x: 0.5, y: 0.25 };
                
            case this.clothingTypes.PANTS:
            case this.clothingTypes.SHORTS:
            case this.clothingTypes.SKIRT:
                // Prendas inferiores: centradas en la cadera
                return { x: 0.5, y: 0.6 };
                
            case this.clothingTypes.DRESS:
                // Vestidos: centrados en cintura pero más arriba
                return { x: 0.5, y: 0.4 };
                
            case this.clothingTypes.HAT:
                // Sombreros: en la cabeza
                return { x: 0.5, y: 0.05 };
                
            case this.clothingTypes.SHOES:
                // Zapatos: en los pies
                return { x: 0.5, y: 0.95 };
                
            default:
                // Posición por defecto
                return { x: 0.5, y: 0.5 };
        }
    }
    
    /**
     * Calcula datos de ajuste para una prenda
     * @param {Object} clothing - Información de la prenda
     * @returns {Object} - Datos de ajuste
     */
    calculateFitData(clothing) {
        // Si no hay medidas, no podemos calcular el ajuste
        if (!this.measurements) {
            return { quality: 'unknown', fit: 'unknown', areas: {} };
        }
        
        // Determinar tipo de prenda
        const type = this.determineClothingType(clothing);
        
        // Talla seleccionada
        const selectedSize = clothing.selectedSize || 'M';
        
        // Objeto para guardar resultados
        const fitData = {
            quality: 'good', // good, tight, loose
            fit: 'regular',  // regular, slim, oversized
            areas: {}        // ajuste por áreas específicas
        };
        
        // Calcular ajuste según tipo de prenda
        switch (type) {
            case this.clothingTypes.SHIRT:
            case this.clothingTypes.TSHIRT:
                this.calculateTopFit(clothing, selectedSize, fitData);
                break;
                
            case this.clothingTypes.PANTS:
            case this.clothingTypes.SHORTS:
                this.calculateBottomFit(clothing, selectedSize, fitData);
                break;
                
            case this.clothingTypes.DRESS:
                this.calculateDressFit(clothing, selectedSize, fitData);
                break;
                
            default:
                // Para otros tipos, usar ajuste genérico
                fitData.quality = 'unknown';
                fitData.fit = 'regular';
                break;
        }
        
        return fitData;
    }
    
    /**
     * Calcula el ajuste para prendas superiores
     * @param {Object} clothing - Información de la prenda
     * @param {string} size - Talla seleccionada
     * @param {Object} fitData - Objeto de resultados
     */
    calculateTopFit(clothing, size, fitData) {
        // Áreas principales para prendas superiores
        const areas = ['chest', 'shoulder', 'sleeve'];
        let tightAreas = 0;
        let looseAreas = 0;
        
        // Verificar cada área
        areas.forEach(area => {
            const userMeasurement = this.measurements[area] || 0;
            const clothingMeasurement = this.getClothingSizeMeasurement(clothing, size, area) || 0;
            
            if (userMeasurement > 0 && clothingMeasurement > 0) {
                // Calcular diferencia porcentual
                const diff = (clothingMeasurement - userMeasurement) / userMeasurement * 100;
                
                // Interpretar la diferencia
                let areaFit = 'regular';
                if (diff < -5) {
                    areaFit = 'tight';
                    tightAreas++;
                } else if (diff > 10) {
                    areaFit = 'loose';
                    looseAreas++;
                }
                
                // Guardar datos del área
                fitData.areas[area] = {
                    user: userMeasurement,
                    clothing: clothingMeasurement,
                    difference: diff.toFixed(1),
                    fit: areaFit
                };
            }
        });
        
        // Determinar calidad general del ajuste
        if (tightAreas > 1) {
            fitData.quality = 'tight';
        } else if (looseAreas > 1) {
            fitData.quality = 'loose';
        } else {
            fitData.quality = 'good';
        }
        
        // Determinar tipo de fit
        if (fitData.areas.chest && fitData.areas.chest.fit === 'tight') {
            fitData.fit = 'slim';
        } else if (fitData.areas.chest && fitData.areas.chest.fit === 'loose') {
            fitData.fit = 'oversized';
        } else {
            fitData.fit = 'regular';
        }
    }
    
    /**
     * Calcula el ajuste para prendas inferiores
     * @param {Object} clothing - Información de la prenda
     * @param {string} size - Talla seleccionada
     * @param {Object} fitData - Objeto de resultados
     */
    calculateBottomFit(clothing, size, fitData) {
        // Áreas principales para prendas inferiores
        const areas = ['waist', 'hips', 'inseam'];
        let tightAreas = 0;
        let looseAreas = 0;
        
        // Verificar cada área
        areas.forEach(area => {
            const userMeasurement = this.measurements[area] || 0;
            const clothingMeasurement = this.getClothingSizeMeasurement(clothing, size, area) || 0;
            
            if (userMeasurement > 0 && clothingMeasurement > 0) {
                // Calcular diferencia porcentual
                const diff = (clothingMeasurement - userMeasurement) / userMeasurement * 100;
                
                // Interpretar la diferencia
                let areaFit = 'regular';
                if (diff < -3) {
                    areaFit = 'tight';
                    tightAreas++;
                } else if (diff > 7) {
                    areaFit = 'loose';
                    looseAreas++;
                }
                
                // Guardar datos del área
                fitData.areas[area] = {
                    user: userMeasurement,
                    clothing: clothingMeasurement,
                    difference: diff.toFixed(1),
                    fit: areaFit
                };
            }
        });
        
        // Determinar calidad general del ajuste
        if (tightAreas > 1) {
            fitData.quality = 'tight';
        } else if (looseAreas > 1) {
            fitData.quality = 'loose';
        } else {
            fitData.quality = 'good';
        }
        
        // Determinar tipo de fit
        if (fitData.areas.hips && fitData.areas.hips.fit === 'tight') {
            fitData.fit = 'slim';
        } else if (fitData.areas.hips && fitData.areas.hips.fit === 'loose') {
            fitData.fit = 'relaxed';
        } else {
            fitData.fit = 'regular';
        }
    }
    
    /**
     * Calcula el ajuste para vestidos
     * @param {Object} clothing - Información de la prenda
     * @param {string} size - Talla seleccionada
     * @param {Object} fitData - Objeto de resultados
     */
    calculateDressFit(clothing, size, fitData) {
        // Combinar aspectos de prendas superiores e inferiores
        this.calculateTopFit(clothing, size, fitData);
        
        // Añadir áreas específicas de vestidos
        const areas = ['waist', 'hips', 'length'];
        let tightAreas = 0;
        let looseAreas = 0;
        
        // Verificar cada área
        areas.forEach(area => {
            if (!fitData.areas[area]) { // Solo procesar si no se ha procesado ya
                const userMeasurement = this.measurements[area] || 0;
                const clothingMeasurement = this.getClothingSizeMeasurement(clothing, size, area) || 0;
                
                if (userMeasurement > 0 && clothingMeasurement > 0) {
                    // Calcular diferencia porcentual
                    const diff = (clothingMeasurement - userMeasurement) / userMeasurement * 100;
                    
                    // Interpretar la diferencia
                    let areaFit = 'regular';
                    if (diff < -4) {
                        areaFit = 'tight';
                        tightAreas++;
                    } else if (diff > 8) {
                        areaFit = 'loose';
                        looseAreas++;
                    }
                    
                    // Guardar datos del área
                    fitData.areas[area] = {
                        user: userMeasurement,
                        clothing: clothingMeasurement,
                        difference: diff.toFixed(1),
                        fit: areaFit
                    };
                }
            }
        });
        
        // Reevaluar calidad general considerando todas las áreas
        const totalTightAreas = Object.values(fitData.areas).filter(a => a.fit === 'tight').length;
        const totalLooseAreas = Object.values(fitData.areas).filter(a => a.fit === 'loose').length;
        
        if (totalTightAreas > 2) {
            fitData.quality = 'tight';
        } else if (totalLooseAreas > 2) {
            fitData.quality = 'loose';
        } else {
            fitData.quality = 'good';
        }
    }
    
    /**
     * Actualiza la etiqueta de talla
     * @param {Object} clothing - Información de la prenda
     */
    updateSizeLabel(clothing) {
        if (!this.sizeLabel) return;
        
        const selectedSize = clothing.selectedSize || 'M';
        
        // Obtener datos de ajuste
        const fitData = this.calculateFitData(clothing);
        
        // Construir HTML para la etiqueta
        let html = `<div class="size-value ${fitData.quality}">${selectedSize}</div>`;
        
        // Añadir descripción del ajuste
        let fitDescription = '';
        switch (fitData.quality) {
            case 'tight':
                fitDescription = 'Esta talla te queda ajustada';
                break;
            case 'loose':
                fitDescription = 'Esta talla te queda holgada';
                break;
            case 'good':
                fitDescription = 'Esta talla te queda bien';
                break;
            default:
                fitDescription = 'Ajuste desconocido';
        }
        
        html += `<div class="fit-description">${fitDescription}</div>`;
        
        // Actualizar la etiqueta
        this.sizeLabel.innerHTML = html;
        this.sizeLabel.style.display = 'block';
    }
    
    /**
     * Muestra u oculta el indicador de carga
     * @param {boolean} show - Mostrar u ocultar
     */
    showLoading(show) {
        this.loading = show;
        
        if (this.loadingLabel) {
            this.loadingLabel.style.display = show ? 'block' : 'none';
        }
    }
    
    /**
     * Renderiza el estado actual
     */
    render() {
        if (!this.ctx || !this.canvas) {
            return;
        }
        
        // Limpiar canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Dibujar fondo
        this.drawBackground();
        
        // Dibujar silueta/cuerpo
        this.drawBody();
        
        // Dibujar prendas por capas
        this.drawClothingLayers();
        
        // Dibujar indicadores si están activados
        if (this.options.showSizeIndicators) {
            this.drawSizeIndicators();
        }
    }
    
    /**
     * Dibuja el fondo del canvas
     */
    drawBackground() {
        // Fondo blanco/gris claro
        this.ctx.fillStyle = '#f8f8f8';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Añadir sombra sutil
        this.ctx.fillStyle = 'rgba(0,0,0,0.05)';
        this.ctx.beginPath();
        this.ctx.ellipse(
            this.canvas.width / 2, 
            this.canvas.height * 0.95, 
            this.canvas.width * 0.3, 
            this.canvas.height * 0.05, 
            0, 0, Math.PI * 2
        );
        this.ctx.fill();
    }
    
    /**
     * Dibuja el cuerpo/silueta
     */
    drawBody() {
        if (this.bodyImage) {
            // Si tenemos una imagen de cuerpo, usarla
            const aspectRatio = this.bodyImage.width / this.bodyImage.height;
            const height = this.canvas.height * 0.9;
            const width = height * aspectRatio;
            
            this.ctx.globalAlpha = 0.7; // Semi-transparente
            this.ctx.drawImage(
                this.bodyImage,
                (this.canvas.width - width) / 2,
                (this.canvas.height - height) / 2,
                width,
                height
            );
            this.ctx.globalAlpha = 1.0;
        } else if (this.options.showBodyOutline && this.bodyLandmarks) {
            // Si no hay imagen pero hay landmarks, dibujar silueta
            this.drawBodyOutline();
        }
        
        // Dibujar landmarks si está activada la opción
        if (this.options.showLandmarks && this.bodyLandmarks) {
            this.drawBodyLandmarks();
        }
    }
    
    /**
     * Dibuja una silueta corporal basada en landmarks
     */
    drawBodyOutline() {
        if (!this.bodyLandmarks || this.bodyLandmarks.length < 10) {
            return;
        }
        
        // Color de relleno para la silueta
        this.ctx.fillStyle = 'rgba(200, 200, 200, 0.3)';
        this.ctx.strokeStyle = 'rgba(180, 180, 180, 0.5)';
        this.ctx.lineWidth = 1;
        
        // Convertir landmarks normalizados a coordenadas de canvas
        const landmarks = this.bodyLandmarks.map(lm => ({
            x: lm.x * this.canvas.width,
            y: lm.y * this.canvas.height
        }));
        
        // Dibujar torso
        this.ctx.beginPath();
        this.ctx.moveTo(landmarks[1].x, landmarks[1].y); // Hombro izquierdo
        this.ctx.lineTo(landmarks[2].x, landmarks[2].y); // Hombro derecho
        this.ctx.lineTo(landmarks[8].x, landmarks[8].y); // Cadera derecha
        this.ctx.lineTo(landmarks[7].x, landmarks[7].y); // Cadera izquierda
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.stroke();
        
        // Dibujar cabeza (círculo simple)
        const headCenter = {
            x: (landmarks[1].x + landmarks[2].x) / 2,
            y: landmarks[1].y - (landmarks[2].x - landmarks[1].x) * 0.4
        };
        const headRadius = (landmarks[2].x - landmarks[1].x) * 0.3;
        
        this.ctx.beginPath();
        this.ctx.arc(headCenter.x, headCenter.y, headRadius, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.stroke();
        
        // Dibujar brazos
        // Brazo izquierdo
        this.ctx.beginPath();
        this.ctx.moveTo(landmarks[1].x, landmarks[1].y); // Hombro izquierdo
        this.ctx.lineTo(landmarks[3].x, landmarks[3].y); // Codo izquierdo
        this.ctx.lineTo(landmarks[5].x, landmarks[5].y); // Muñeca izquierda
        this.ctx.stroke();
        
        // Brazo derecho
        this.ctx.beginPath();
        this.ctx.moveTo(landmarks[2].x, landmarks[2].y); // Hombro derecho
        this.ctx.lineTo(landmarks[4].x, landmarks[4].y); // Codo derecho
        this.ctx.lineTo(landmarks[6].x, landmarks[6].y); // Muñeca derecha
        this.ctx.stroke();
        
        // Dibujar piernas
        // Pierna izquierda
        this.ctx.beginPath();
        this.ctx.moveTo(landmarks[7].x, landmarks[7].y); // Cadera izquierda
        this.ctx.lineTo(landmarks[9].x, landmarks[9].y); // Rodilla izquierda
        this.ctx.lineTo(landmarks[11].x, landmarks[11].y); // Tobillo izquierdo
        this.ctx.stroke();
        
        // Pierna derecha
        this.ctx.beginPath();
        this.ctx.moveTo(landmarks[8].x, landmarks[8].y); // Cadera derecha
        this.ctx.lineTo(landmarks[10].x, landmarks[10].y); // Rodilla derecha
        this.ctx.lineTo(landmarks[12].x, landmarks[12].y); // Tobillo derecho
        this.ctx.stroke();
    }
    
    /**
     * Dibuja los landmarks corporales
     */
    drawBodyLandmarks() {
        if (!this.bodyLandmarks) {
            return;
        }
        
        // Configuración para dibujar landmarks
        this.ctx.fillStyle = 'rgba(0, 150, 255, 0.7)';
        
        // Dibujar cada landmark
        this.bodyLandmarks.forEach(landmark => {
            const x = landmark.x * this.canvas.width;
            const y = landmark.y * this.canvas.height;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, 4, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }
    
    /**
     * Dibuja las capas de prendas
     */
    drawClothingLayers() {
        // Orden de capas (de abajo hacia arriba)
        const layerOrder = ['bottom', 'middle', 'full', 'top', 'head', 'feet'];
        
        // Dibujar cada capa en orden
        layerOrder.forEach(layer => {
            if (this.clothingLayers[layer]) {
                this.drawClothing(this.clothingLayers[layer]);
            }
        });
    }
    
    /**
     * Dibuja una prenda específica
     * @param {Object} clothing - Prenda a dibujar
     */
    drawClothing(clothing) {
        if (!clothing || !clothing.image) {
            return;
        }
        
        // Obtener transformación
        const transform = clothing.transformation;
        
        // Calcular dimensiones
        const aspectRatio = clothing.image.width / clothing.image.height;
        let width, height;
        
        // Ajustar según tipo de prenda
        switch (clothing.type) {
            case this.clothingTypes.SHIRT:
            case this.clothingTypes.TSHIRT:
            case this.clothingTypes.JACKET:
            case this.clothingTypes.SWEATER:
                // Prendas superiores: ~40% del alto del canvas
                height = this.canvas.height * 0.4;
                width = height * aspectRatio;
                break;
                
            case this.clothingTypes.PANTS:
            case this.clothingTypes.SKIRT:
                // Prendas inferiores: ~50% del alto del canvas
                height = this.canvas.height * 0.5;
                width = height * aspectRatio;
                break;
                
            case this.clothingTypes.SHORTS:
                // Shorts: ~30% del alto del canvas
                height = this.canvas.height * 0.3;
                width = height * aspectRatio;
                break;
                
            case this.clothingTypes.DRESS:
                // Vestidos: ~75% del alto del canvas
                height = this.canvas.height * 0.75;
                width = height * aspectRatio;
                break;
                
            default:
                // Por defecto: 50% del alto del canvas
                height = this.canvas.height * 0.5;
                width = height * aspectRatio;
        }
        
        // Aplicar escala
        width *= transform.scale;
        height *= transform.scale;
        
        // Calcular posición (centrada en el punto de referencia)
        const x = transform.position.x * this.canvas.width - width / 2;
        const y = transform.position.y * this.canvas.height - height / 2;
        
        // Dibujar la prenda
        this.ctx.globalAlpha = transform.opacity;
        
        if (transform.rotation !== 0) {
            // Si hay rotación, aplicar transformación
            this.ctx.save();
            this.ctx.translate(
                transform.position.x * this.canvas.width,
                transform.position.y * this.canvas.height
            );
            this.ctx.rotate(transform.rotation * Math.PI / 180);
            this.ctx.drawImage(clothing.image, -width / 2, -height / 2, width, height);
            this.ctx.restore();
        } else {
            // Sin rotación, dibujar directamente
            this.ctx.drawImage(clothing.image, x, y, width, height);
        }
        
        this.ctx.globalAlpha = 1.0;
    }
    
    /**
     * Dibuja indicadores de talla
     */
    drawSizeIndicators() {
        // Solo dibujar si hay una prenda actual y datos de ajuste
        if (!this.currentClothing || !this.clothingLayers) {
            return;
        }
        
        // Buscar la prenda activa en las capas
        const activeLayers = Object.values(this.clothingLayers).filter(layer => 
            layer.info && layer.info.id === this.currentClothing.id);
        
        if (activeLayers.length === 0) {
            return;
        }
        
        const activeLayer = activeLayers[0];
        
        // Solo dibujar si tiene datos de ajuste
        if (!activeLayer.fit || !activeLayer.fit.areas) {
            return;
        }
        
        // Configuración para indicadores
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        
        // Dibujar indicadores según tipo de prenda
        switch (activeLayer.type) {
            case this.clothingTypes.SHIRT:
            case this.clothingTypes.TSHIRT:
                this.drawTopSizeIndicators(activeLayer);
                break;
                
            case this.clothingTypes.PANTS:
            case this.clothingTypes.SHORTS:
                this.drawBottomSizeIndicators(activeLayer);
                break;
                
            case this.clothingTypes.DRESS:
                this.drawTopSizeIndicators(activeLayer);
                this.drawBottomSizeIndicators(activeLayer);
                break;
        }
    }
    
    /**
     * Dibuja indicadores de talla para prendas superiores
     * @param {Object} clothing - Prenda superior
     */
    drawTopSizeIndicators(clothing) {
        if (!clothing.fit || !clothing.fit.areas) {
            return;
        }
        
        const areas = clothing.fit.areas;
        
        // Convertir landmarks normalizados a coordenadas de canvas
        const landmarks = this.bodyLandmarks.map(lm => ({
            x: lm.x * this.canvas.width,
            y: lm.y * this.canvas.height
        }));
        
        // Indicador de pecho (entre hombros)
        if (areas.chest) {
            const chestY = (landmarks[1].y + landmarks[7].y) / 2.5;
            const chestX = this.canvas.width / 2;
            
            this.drawFitIndicator(
                chestX, chestY, 
                'Pecho', 
                areas.chest.difference, 
                areas.chest.fit
            );
        }
        
        // Indicador de hombros
        if (areas.shoulder) {
            const shoulderY = landmarks[1].y - 15;
            const shoulderX = (landmarks[1].x + landmarks[2].x) / 2;
            
            this.drawFitIndicator(
                shoulderX, shoulderY, 
                'Hombros', 
                areas.shoulder.difference, 
                areas.shoulder.fit
            );
        }
    }
    
    /**
     * Dibuja indicadores de talla para prendas inferiores
     * @param {Object} clothing - Prenda inferior
     */
    drawBottomSizeIndicators(clothing) {
        if (!clothing.fit || !clothing.fit.areas) {
            return;
        }
        
        const areas = clothing.fit.areas;
        
        // Convertir landmarks normalizados a coordenadas de canvas
        const landmarks = this.bodyLandmarks.map(lm => ({
            x: lm.x * this.canvas.width,
            y: lm.y * this.canvas.height
        }));
        
        // Indicador de cintura
        if (areas.waist) {
            const waistY = (landmarks[7].y + landmarks[8].y) / 2 - 15;
            const waistX = (landmarks[7].x + landmarks[8].x) / 2;
            
            this.drawFitIndicator(
                waistX, waistY, 
                'Cintura', 
                areas.waist.difference, 
                areas.waist.fit
            );
        }
        
        // Indicador de cadera
        if (areas.hips) {
            const hipsY = (landmarks[7].y + landmarks[8].y) / 2 + 30;
            const hipsX = (landmarks[7].x + landmarks[8].x) / 2;
            
            this.drawFitIndicator(
                hipsX, hipsY, 
                'Cadera', 
                areas.hips.difference, 
                areas.hips.fit
            );
        }
    }
    
    /**
     * Dibuja un indicador de ajuste
     * @param {number} x - Posición X
     * @param {number} y - Posición Y
     * @param {string} label - Etiqueta
     * @param {string|number} difference - Diferencia
     * @param {string} fit - Tipo de ajuste
     */
    drawFitIndicator(x, y, label, difference, fit) {
        // Determinar color según ajuste
        let color;
        switch (fit) {
            case 'tight':
                color = 'rgba(255, 80, 80, 0.8)';
                break;
            case 'loose':
                color = 'rgba(80, 80, 255, 0.8)';
                break;
            default:
                color = 'rgba(80, 200, 80, 0.8)';
        }
        
        // Dibujar línea al punto
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(x, y - 30);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
        
        // Dibujar círculo en el punto
        this.ctx.beginPath();
        this.ctx.arc(x, y, 4, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        
        // Dibujar fondo para texto
        const textWidth = label.length * 6 + 30;
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.fillRect(x - textWidth / 2, y - 45, textWidth, 25);
        
        // Dibujar borde
        this.ctx.strokeStyle = color;
        this.ctx.strokeRect(x - textWidth / 2, y - 45, textWidth, 25);
        
        // Dibujar texto
        this.ctx.fillStyle = '#333';
        this.ctx.font = 'bold 12px Arial';
        this.ctx.fillText(label, x, y - 32);
        
        // Dibujar diferencia si existe
        if (difference) {
            this.ctx.font = '10px Arial';
            
            // Formatear diferencia
            let diffText;
            if (typeof difference === 'number') {
                diffText = difference > 0 ? `+${difference.toFixed(1)}%` : `${difference.toFixed(1)}%`;
            } else {
                diffText = difference.startsWith('-') ? difference : `+${difference}`;
            }
            
            this.ctx.fillText(diffText, x, y - 20);
        }
    }
    
    /**
     * Genera una captura del estado actual
     * @returns {Promise<string>} - URL de la imagen
     */
    async capturePreview() {
        return new Promise((resolve) => {
            // Crear captura del canvas actual
            const dataUrl = this.canvas.toDataURL('image/png');
            resolve(dataUrl);
        });
    }
    
    /**
     * Configura el callback para cuando se completa el ajuste
     * @param {Function} callback - Función de callback
     */
    onFitting(callback) {
        if (typeof callback === 'function') {
            this.onFittingComplete = callback;
        }
    }
    
    /**
     * Actualiza las opciones de configuración
     * @param {Object} newOptions - Nuevas opciones
     */
    updateOptions(newOptions) {
        // Actualizar opciones
        this.options = {...this.options, ...newOptions};
        
        // Re-renderizar con las nuevas opciones
        this.render();
    }
    
    /**
     * Limpia el estado actual
     */
    clear() {
        // Limpiar prendas
        this.clothingLayers = {};
        this.currentClothing = null;
        
        // Ocultar etiqueta de talla
        if (this.sizeLabel) {
            this.sizeLabel.style.display = 'none';
        }
        
        // Re-renderizar
        this.render();
    }
    
    /**
     * Destruye el ajustador de prendas y libera recursos
     */
    dispose() {
        // Limpiar canvas
        if (this.ctx && this.canvas) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
        
        // Eliminar elementos UI
        if (this.container) {
            if (this.controlsContainer) {
                this.container.removeChild(this.controlsContainer);
            }
            
            if (this.loadingLabel) {
                this.container.removeChild(this.loadingLabel);
            }
            
            if (this.sizeLabel) {
                this.container.removeChild(this.sizeLabel);
            }
            
            if (this.canvas) {
                this.container.removeChild(this.canvas);
            }
        }
        
        // Limpiar referencias
        this.clothingLayers = {};
        this.currentClothing = null;
        this.bodyImage = null;
        this.bodyLandmarks = null;
        this.isInitialized = false;
    }
}

// Exportar la clase para uso en otros módulos
export default ClothingFitter;
