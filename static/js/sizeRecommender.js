/**
 * sizeRecommender.js - Módulo para recomendar tallas de ropa
 * 
 * Este módulo se encarga de:
 * - Analizar las medidas del usuario
 * - Comparar con tablas de tallas de diferentes marcas
 * - Calcular la talla más adecuada para el usuario
 * - Mostrar recomendaciones de talla personalizadas
 */

/**
 * Clase principal para recomendación de tallas
 */
class SizeRecommender {
    /**
     * Constructor
     * @param {Object} options - Opciones de configuración
     * @param {HTMLElement} options.container - Contenedor para mostrar recomendaciones
     * @param {Object} options.measurements - Medidas del usuario
     */
    constructor(options = {}) {
        // Elementos DOM
        this.container = options.container || null;
        
        // Medidas del usuario
        this.measurements = options.measurements || null;
        
        // Estado
        this.isInitialized = false;
        this.loading = false;
        this.currentBrand = null;
        this.selectedClothing = null;
        this.sizeCharts = {};
        this.userPreferences = {
            fit: 'regular', // regular, slim, loose
            region: 'EU'    // EU, US, UK, INT
        };
        
        // Referencias a elementos DOM
        this.recommendationElement = null;
        this.alternativesElement = null;
        this.fitInfoElement = null;
        this.fitExplanationElement = null;
        this.brandSelectorElement = null;
        this.fitPreferenceElement = null;
        
        // Inicialización automática si hay un contenedor
        if (this.container) {
            this.initialize();
        }
    }
    
    /**
     * Inicializa el recomendador de tallas
     * @returns {Promise<boolean>} - Éxito de la inicialización
     */
    async initialize() {
        if (this.isInitialized) {
            return true;
        }
        
        try {
            console.log("Inicializando recomendador de tallas...");
            
            // Crear estructura UI
            this.createUI();
            
            // Cargar tablas de tallas
            await this.loadSizeCharts();
            
            this.isInitialized = true;
            console.log("Recomendador de tallas inicializado correctamente");
            
            return true;
        } catch (error) {
            console.error("Error al inicializar el recomendador de tallas:", error);
            return false;
        }
    }
    
    /**
     * Crea la estructura de UI
     */
    createUI() {
        // Verificar contenedor
        if (!this.container) {
            console.error("No se ha proporcionado un contenedor para el recomendador de tallas");
            return;
        }
        
        // Contenedor principal
        this.container.innerHTML = '';
        this.container.className = 'size-recommendation-container';
        
        // Título
        const title = document.createElement('h3');
        title.className = 'recommendation-title';
        title.textContent = 'Recomendación de talla';
        this.container.appendChild(title);
        
        // Selector de marca
        this.createBrandSelector();
        
        // Selector de preferencia de ajuste
        this.createFitPreferenceSelector();
        
        // Información de talla recomendada
        this.recommendationElement = document.createElement('div');
        this.recommendationElement.className = 'size-recommendation';
        this.container.appendChild(this.recommendationElement);
        
        // Tallas alternativas
        this.alternativesElement = document.createElement('div');
        this.alternativesElement.className = 'size-alternatives';
        this.container.appendChild(this.alternativesElement);
        
        // Información de ajuste
        this.fitInfoElement = document.createElement('div');
        this.fitInfoElement.className = 'fit-info';
        this.container.appendChild(this.fitInfoElement);
        
        // Explicación detallada del ajuste
        this.fitExplanationElement = document.createElement('div');
        this.fitExplanationElement.className = 'fit-explanation';
        this.container.appendChild(this.fitExplanationElement);
        
        // Estado inicial (placeholder)
        this.showPlaceholder();
    }
    
    /**
     * Crea el selector de marcas
     */
    createBrandSelector() {
        // Contenedor
        const selectorContainer = document.createElement('div');
        selectorContainer.className = 'brand-selector-container';
        
        // Etiqueta
        const label = document.createElement('label');
        label.textContent = 'Marca:';
        label.setAttribute('for', 'brand-selector');
        selectorContainer.appendChild(label);
        
        // Selector
        this.brandSelectorElement = document.createElement('select');
        this.brandSelectorElement.id = 'brand-selector';
        this.brandSelectorElement.className = 'brand-selector';
        
        // Opción por defecto
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Seleccionar marca';
        defaultOption.disabled = true;
        defaultOption.selected = true;
        this.brandSelectorElement.appendChild(defaultOption);
        
        // Marcas comunes
        const commonBrands = [
            'generic', 'zara', 'hm', 'levis', 'nike', 'adidas', 'gap', 
            'uniqlo', 'mango', 'bershka', 'pullbear', 'tommy'
        ];
        
        commonBrands.forEach(brand => {
            const option = document.createElement('option');
            option.value = brand;
            option.textContent = brand === 'generic' ? 'Genérica' : brand.charAt(0).toUpperCase() + brand.slice(1);
            this.brandSelectorElement.appendChild(option);
        });
        
        // Evento de cambio
        this.brandSelectorElement.addEventListener('change', () => {
            this.currentBrand = this.brandSelectorElement.value;
            if (this.selectedClothing) {
                this.recommendSize(this.selectedClothing);
            }
        });
        
        selectorContainer.appendChild(this.brandSelectorElement);
        this.container.appendChild(selectorContainer);
    }
    
    /**
     * Crea el selector de preferencia de ajuste
     */
    createFitPreferenceSelector() {
        // Contenedor
        const selectorContainer = document.createElement('div');
        selectorContainer.className = 'fit-preference-container';
        
        // Etiqueta
        const label = document.createElement('label');
        label.textContent = 'Preferencia de ajuste:';
        label.setAttribute('for', 'fit-preference');
        selectorContainer.appendChild(label);
        
        // Selector
        this.fitPreferenceElement = document.createElement('select');
        this.fitPreferenceElement.id = 'fit-preference';
        this.fitPreferenceElement.className = 'fit-preference';
        
        // Opciones
        const fitOptions = [
            { value: 'slim', label: 'Ajustado' },
            { value: 'regular', label: 'Regular' },
            { value: 'loose', label: 'Holgado' }
        ];
        
        fitOptions.forEach(option => {
            const optElement = document.createElement('option');
            optElement.value = option.value;
            optElement.textContent = option.label;
            if (option.value === this.userPreferences.fit) {
                optElement.selected = true;
            }
            this.fitPreferenceElement.appendChild(optElement);
        });
        
        // Evento de cambio
        this.fitPreferenceElement.addEventListener('change', () => {
            this.userPreferences.fit = this.fitPreferenceElement.value;
            if (this.selectedClothing) {
                this.recommendSize(this.selectedClothing);
            }
        });
        
        selectorContainer.appendChild(this.fitPreferenceElement);
        this.container.appendChild(selectorContainer);
    }
    
    /**
     * Muestra un placeholder inicial
     */
    showPlaceholder() {
        if (!this.recommendationElement) return;
        
        this.recommendationElement.innerHTML = `
            <div class="placeholder-message">
                <i class="fa fa-tshirt"></i>
                <p>Selecciona una prenda para ver la recomendación de talla</p>
            </div>
        `;
        
        this.alternativesElement.innerHTML = '';
        this.fitInfoElement.innerHTML = '';
        this.fitExplanationElement.innerHTML = '';
    }
    
    /**
     * Carga las tablas de tallas desde el servidor
     * @returns {Promise<boolean>} - Éxito de la carga
     */
    async loadSizeCharts() {
        try {
            this.loading = true;
            
            // Cargar tablas genéricas
            const genericMaleResponse = await fetch('/data/references/size_charts/generic_male.json');
            const genericFemaleResponse = await fetch('/data/references/size_charts/generic_female.json');
            
            if (genericMaleResponse.ok && genericFemaleResponse.ok) {
                const genericMale = await genericMaleResponse.json();
                const genericFemale = await genericFemaleResponse.json();
                
                this.sizeCharts.generic = {
                    male: genericMale,
                    female: genericFemale
                };
            }
            
            // Cargar tablas de marcas específicas
            const brands = ['zara', 'hm', 'levis'];
            
            for (const brand of brands) {
                const response = await fetch(`/data/references/size_charts/${brand}.json`);
                if (response.ok) {
                    this.sizeCharts[brand] = await response.json();
                }
            }
            
            this.loading = false;
            return true;
        } catch (error) {
            console.error("Error al cargar tablas de tallas:", error);
            this.loading = false;
            
            // Si fallan las tablas específicas, usar las genéricas como fallback
            if (!this.sizeCharts.generic) {
                this.sizeCharts.generic = this.getDefaultSizeCharts();
            }
            
            return false;
        }
    }
    
    /**
     * Obtiene tablas de tallas por defecto
     * @returns {Object} - Tablas de tallas por defecto
     */
    getDefaultSizeCharts() {
        return {
            male: {
                tops: {
                    measurements: ["chest", "waist", "hips", "shoulder"],
                    sizes: {
                        "XS": { chest: [84, 89], waist: [71, 76], hips: [86, 91], shoulder: [40, 42] },
                        "S":  { chest: [90, 95], waist: [77, 82], hips: [92, 97], shoulder: [43, 45] },
                        "M":  { chest: [96, 101], waist: [83, 88], hips: [98, 103], shoulder: [46, 48] },
                        "L":  { chest: [102, 107], waist: [89, 94], hips: [104, 109], shoulder: [49, 51] },
                        "XL": { chest: [108, 113], waist: [95, 100], hips: [110, 115], shoulder: [52, 54] },
                        "XXL": { chest: [114, 119], waist: [101, 106], hips: [116, 121], shoulder: [55, 57] }
                    }
                },
                bottoms: {
                    measurements: ["waist", "hips", "inseam"],
                    sizes: {
                        "28": { waist: [71, 73], hips: [86, 88], inseam: [78, 79] },
                        "30": { waist: [76, 78], hips: [91, 93], inseam: [79, 80] },
                        "32": { waist: [81, 83], hips: [96, 98], inseam: [80, 81] },
                        "34": { waist: [86, 88], hips: [101, 103], inseam: [81, 82] },
                        "36": { waist: [91, 93], hips: [106, 108], inseam: [82, 83] },
                        "38": { waist: [96, 98], hips: [111, 113], inseam: [83, 84] }
                    }
                }
            },
            female: {
                tops: {
                    measurements: ["chest", "waist", "hips", "shoulder"],
                    sizes: {
                        "XS": { chest: [80, 85], waist: [60, 65], hips: [86, 91], shoulder: [37, 39] },
                        "S":  { chest: [86, 91], waist: [66, 71], hips: [92, 97], shoulder: [39, 41] },
                        "M":  { chest: [92, 97], waist: [72, 77], hips: [98, 103], shoulder: [41, 43] },
                        "L":  { chest: [98, 103], waist: [78, 83], hips: [104, 109], shoulder: [43, 45] },
                        "XL": { chest: [104, 109], waist: [84, 89], hips: [110, 115], shoulder: [45, 47] },
                        "XXL": { chest: [110, 115], waist: [90, 95], hips: [116, 121], shoulder: [47, 49] }
                    }
                },
                bottoms: {
                    measurements: ["waist", "hips", "inseam"],
                    sizes: {
                        "XS / 34": { waist: [60, 65], hips: [86, 91], inseam: [76, 77] },
                        "S / 36":  { waist: [66, 71], hips: [92, 97], inseam: [77, 78] },
                        "M / 38":  { waist: [72, 77], hips: [98, 103], inseam: [78, 79] },
                        "L / 40":  { waist: [78, 83], hips: [104, 109], inseam: [79, 80] },
                        "XL / 42": { waist: [84, 89], hips: [110, 115], inseam: [80, 81] },
                        "XXL / 44": { waist: [90, 95], hips: [116, 121], inseam: [81, 82] }
                    }
                },
                dresses: {
                    measurements: ["chest", "waist", "hips"],
                    sizes: {
                        "XS / 34": { chest: [80, 85], waist: [60, 65], hips: [86, 91] },
                        "S / 36":  { chest: [86, 91], waist: [66, 71], hips: [92, 97] },
                        "M / 38":  { chest: [92, 97], waist: [72, 77], hips: [98, 103] },
                        "L / 40":  { chest: [98, 103], waist: [78, 83], hips: [104, 109] },
                        "XL / 42": { chest: [104, 109], waist: [84, 89], hips: [110, 115] },
                        "XXL / 44": { chest: [110, 115], waist: [90, 95], hips: [116, 121] }
                    }
                }
            }
        };
    }
    
    /**
     * Establece las medidas del usuario
     * @param {Object} measurements - Medidas del usuario
     */
    setMeasurements(measurements) {
        this.measurements = measurements;
        
        // Si hay una prenda seleccionada, recalcular recomendación
        if (this.selectedClothing) {
            this.recommendSize(this.selectedClothing);
        }
    }
    
    /**
     * Recomienda talla para una prenda
     * @param {Object} clothing - Información de la prenda
     * @returns {Promise<Object>} - Resultado de la recomendación
     */
    async recommendSize(clothing) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        if (!this.measurements) {
            this.showNoMeasurementsMessage();
            return null;
        }
        
        try {
            this.loading = true;
            this.selectedClothing = clothing;
            
            // Obtener tipo de prenda
            const clothingType = this.getClothingType(clothing);
            
            // Determinar la marca a usar (la de la prenda, la seleccionada, o genérica)
            const brand = this.determineBrand(clothing);
            
            // Obtener tabla de tallas para esta marca y tipo
            const sizeChart = this.getSizeChart(brand, clothingType);
            if (!sizeChart) {
                throw new Error(`No se encontró tabla de tallas para ${brand} - ${clothingType}`);
            }
            
            // Calcular talla recomendada
            const recommendation = this.calculateRecommendedSize(clothing, sizeChart);
            
            // Mostrar recomendación
            this.displayRecommendation(recommendation, clothing);
            
            this.loading = false;
            return recommendation;
        } catch (error) {
            console.error("Error al recomendar talla:", error);
            this.showError(error.message);
            this.loading = false;
            return null;
        }
    }
    
    /**
     * Muestra mensaje de error por falta de medidas
     */
    showNoMeasurementsMessage() {
        if (!this.recommendationElement) return;
        
        this.recommendationElement.innerHTML = `
            <div class="error-message">
                <i class="fa fa-exclamation-circle"></i>
                <p>No hay medidas disponibles. Por favor, completa el proceso de medición.</p>
                <a href="/measurement" class="action-button">Ir a medir</a>
            </div>
        `;
        
        this.alternativesElement.innerHTML = '';
        this.fitInfoElement.innerHTML = '';
        this.fitExplanationElement.innerHTML = '';
    }
    
    /**
     * Muestra un mensaje de error
     * @param {string} message - Mensaje de error
     */
    showError(message) {
        if (!this.recommendationElement) return;
        
        this.recommendationElement.innerHTML = `
            <div class="error-message">
                <i class="fa fa-exclamation-circle"></i>
                <p>${message}</p>
            </div>
        `;
    }
    
    /**
     * Obtiene el tipo de prenda (para buscar la tabla adecuada)
     * @param {Object} clothing - Información de la prenda
     * @returns {string} - Tipo de prenda (tops, bottoms, dresses)
     */
    getClothingType(clothing) {
        // Si la prenda ya tiene un tipo definido, mapearlo a las categorías de tablas
        if (clothing.type) {
            const type = clothing.type.toLowerCase();
            
            if (type.includes('shirt') || type.includes('top') || 
                type.includes('jacket') || type.includes('sweater')) {
                return 'tops';
            } else if (type.includes('pant') || type.includes('trouser') || 
                      type.includes('short') || type.includes('skirt')) {
                return 'bottoms';
            } else if (type.includes('dress')) {
                return 'dresses';
            }
        }
        
        // Intentar determinar por nombre/categoría
        const name = clothing.name ? clothing.name.toLowerCase() : '';
        const category = clothing.category ? clothing.category.toLowerCase() : '';
        
        if (name.includes('camisa') || name.includes('camiseta') || 
            name.includes('top') || name.includes('blusa') || 
            name.includes('polo') || name.includes('sudadera') || 
            name.includes('chaqueta') || name.includes('jacket') || 
            category.includes('superior')) {
            return 'tops';
        } else if (name.includes('pantalón') || name.includes('pantalon') || 
                  name.includes('jeans') || name.includes('short') || 
                  name.includes('bermuda') || name.includes('falda') || 
                  category.includes('inferior')) {
            return 'bottoms';
        } else if (name.includes('vestido') || name.includes('dress') || 
                  category.includes('vestido')) {
            return 'dresses';
        }
        
        // Por defecto, asumir tops
        return 'tops';
    }
    
    /**
     * Determina qué marca usar para la recomendación
     * @param {Object} clothing - Información de la prenda
     * @returns {string} - Marca a usar
     */
    determineBrand(clothing) {
        // Si hay una marca seleccionada manualmente, usarla
        if (this.currentBrand && this.sizeCharts[this.currentBrand]) {
            return this.currentBrand;
        }
        
        // Si la prenda tiene una marca y está disponible en las tablas, usarla
        if (clothing.brand && this.sizeCharts[clothing.brand.toLowerCase()]) {
            return clothing.brand.toLowerCase();
        }
        
        // Usar marca genérica como fallback
        return 'generic';
    }
    
    /**
     * Obtiene la tabla de tallas adecuada
     * @param {string} brand - Marca 
     * @param {string} type - Tipo de prenda
     * @returns {Object|null} - Tabla de tallas
     */
    getSizeChart(brand, type) {
        // Verificar disponibilidad de la tabla
        if (!this.sizeCharts[brand]) {
            // Si no tenemos la marca específica, usar genérica
            brand = 'generic';
            if (!this.sizeCharts[brand]) {
                return null;
            }
        }
        
        // Determinar género (por defecto asumir masculino por simplicidad)
        // En una implementación real, esto debería basarse en el usuario o la prenda
        const gender = this.determineGender();
        
        // Si la marca tiene tablas específicas por género
        if (this.sizeCharts[brand][gender] && this.sizeCharts[brand][gender][type]) {
            return this.sizeCharts[brand][gender][type];
        }
        
        // Si la marca tiene la tabla para el tipo solicitado sin subdivisión por género
        if (this.sizeCharts[brand][type]) {
            return this.sizeCharts[brand][type];
        }
        
        // Si llegamos aquí, intentar con la tabla genérica
        if (brand !== 'generic' && this.sizeCharts.generic) {
            return this.getSizeChart('generic', type);
        }
        
        return null;
    }
    
    /**
     * Determina el género para las tablas
     * @returns {string} - Género ('male' o 'female')
     */
    determineGender() {
        // En una implementación real, esto debería basarse en el perfil del usuario
        // o en metadatos de la prenda. Por simplicidad, aquí asumimos un valor.
        
        // Podría leer del perfil de usuario, o usar la presencia de ciertas medidas
        // como indicador (ej: medidas de busto más grandes podrían indicar femenino)
        
        return 'male'; // Por defecto asumimos masculino
    }
    
    /**
     * Calcula la talla recomendada
     * @param {Object} clothing - Información de la prenda
     * @param {Object} sizeChart - Tabla de tallas
     * @returns {Object} - Resultado de la recomendación
     */
    calculateRecommendedSize(clothing, sizeChart) {
        // Estructura para el resultado
        const result = {
            recommendedSize: null,
            alternatives: [],
            fitScore: 0,  // 0-100, donde 100 es ajuste perfecto
            measurements: {},
            details: {}
        };
        
        // Obtener las medidas relevantes según la tabla
        const relevantMeasurements = sizeChart.measurements || [];
        const userMeasurements = this.extractUserMeasurements(relevantMeasurements);
        
        // Verificar si tenemos suficientes medidas para hacer una recomendación
        const hasEnoughMeasurements = Object.keys(userMeasurements).length > 0;
        if (!hasEnoughMeasurements) {
            throw new Error("No hay suficientes medidas para hacer una recomendación");
        }
        
        // Calcular puntuación para cada talla
        const sizeScores = {};
        
        // Recorrer cada talla en la tabla
        Object.entries(sizeChart.sizes).forEach(([size, sizeMeasurements]) => {
            let totalScore = 0;
            let measurementCount = 0;
            const details = {};
            
            // Evaluar cada medida relevante
            relevantMeasurements.forEach(measurement => {
                if (userMeasurements[measurement] && sizeMeasurements[measurement]) {
                    const userValue = userMeasurements[measurement];
                    const [min, max] = sizeMeasurements[measurement];
                    
                    // Calcular qué tan bien encaja esta medida
                    const fitScore = this.calculateMeasurementFitScore(userValue, min, max);
                    totalScore += fitScore;
                    measurementCount++;
                    
                    // Guardar detalles
                    details[measurement] = {
                        user: userValue,
                        min: min,
                        max: max,
                        score: fitScore,
                        fit: this.getFitDescription(userValue, min, max)
                    };
                }
            });
            
            // Calcular puntuación promedio
            const averageScore = measurementCount > 0 ? totalScore / measurementCount : 0;
            
            // Ajustar según preferencia del usuario
            const adjustedScore = this.adjustScoreForPreference(averageScore, details);
            
            // Guardar puntuación y detalles
            sizeScores[size] = {
                score: adjustedScore,
                details: details
            };
        });
        
        // Ordenar tallas por puntuación
        const sortedSizes = Object.entries(sizeScores)
            .sort((a, b) => b[1].score - a[1].score);
        
        // La primera es la recomendada
        if (sortedSizes.length > 0) {
            result.recommendedSize = sortedSizes[0][0];
            result.fitScore = sortedSizes[0][1].score;
            result.details = sortedSizes[0][1].details;
            
            // Añadir alternativas si hay más tallas
            for (let i = 1; i < sortedSizes.length && i < 3; i++) {
                if (sortedSizes[i][1].score > 70) { // Solo alternativas razonables
                    result.alternatives.push({
                        size: sortedSizes[i][0],
                        score: sortedSizes[i][1].score,
                        details: sortedSizes[i][1].details
                    });
                }
            }
        }
        
        // Incluir medidas relevantes del usuario
        result.measurements = userMeasurements;
        
        return result;
    }
    
    /**
     * Extrae las medidas relevantes del usuario
     * @param {Array} relevantMeasurements - Lista de medidas relevantes
     * @returns {Object} - Medidas del usuario
     */
    extractUserMeasurements(relevantMeasurements) {
        const result = {};
        
        if (!this.measurements || !this.measurements.values) {
            return result;
        }
        
        const values = this.measurements.values;
        
        // Mapeo entre nombres internos y los de la tabla
        const measurementMap = {
            'chest': ['chest', 'chestCircumference', 'bust'],
            'waist': ['waist', 'waistCircumference'],
            'hips': ['hips', 'hipCircumference'],
            'shoulder': ['shoulders', 'shoulderWidth'],
            'inseam': ['inseam', 'insideSeam', 'insideLeg']
        };
        
        // Extraer medidas relevantes
        relevantMeasurements.forEach(measurement => {
            const aliases = measurementMap[measurement] || [measurement];
            
            // Buscar entre posibles alias
            for (const alias of aliases) {
                if (values[alias] !== undefined) {
                    result[measurement] = values[alias];
                    break;
                }
            }
        });
        
        return result;
    }
    
    /**
     * Calcula la puntuación de ajuste para una medida
     * @param {number} userValue - Valor del usuario
     * @param {number} min - Valor mínimo de la talla
     * @param {number} max - Valor máximo de la talla
     * @returns {number} - Puntuación (0-100)
     */
    calculateMeasurementFitScore(userValue, min, max) {
        // Si el valor está dentro del rango, es un ajuste perfecto
        if (userValue >= min && userValue <= max) {
            // Posición relativa dentro del rango (0 a 1)
            const rangePosition = (userValue - min) / (max - min);
            
            // Centro del rango = 100, extremos = 90
            return 90 + 10 * (1 - Math.abs(2 * rangePosition - 1));
        }
        
        // Si está fuera del rango, calcular puntuación basada en distancia
        if (userValue < min) {
            // Demasiado pequeño
            const distance = min - userValue;
            return Math.max(0, 90 - (distance * 10));
        } else {
            // Demasiado grande
            const distance = userValue - max;
            return Math.max(0, 90 - (distance * 10));
        }
    }
    
    /**
     * Ajusta la puntuación según la preferencia del usuario
     * @param {number} score - Puntuación original
     * @param {Object} details - Detalles de ajuste
     * @returns {number} - Puntuación ajustada
     */
    adjustScoreForPreference(score, details) {
        const preference = this.userPreferences.fit;
        let adjustment = 0;
        
        // Contar medidas ajustadas, regulares y holgadas
        let tightCount = 0;
        let regularCount = 0;
        let looseCount = 0;
        
        Object.values(details).forEach(detail => {
            const fit = detail.fit;
            if (fit === 'tight') tightCount++;
            else if (fit === 'regular') regularCount++;
            else if (fit === 'loose') looseCount++;
        });
        
        // Ajustar según preferencia
        switch (preference) {
            case 'slim':
                // Premiar ajuste apretado, penalizar holgado
                adjustment = tightCount * 5 - looseCount * 10;
                break;
            case 'regular':
                // Premiar ajuste regular, penalizar extremos
                adjustment = regularCount * 5 - (tightCount + looseCount) * 5;
                break;
            case 'loose':
                // Premiar ajuste holgado, penalizar apretado
                adjustment = looseCount * 5 - tightCount * 10;
                break;
        }
        
        // Aplicar ajuste y mantener entre 0-100
        return Math.max(0, Math.min(100, score + adjustment));
    }
    
    /**
     * Obtiene una descripción de ajuste
     * @param {number} userValue - Valor del usuario
     * @param {number} min - Valor mínimo de la talla
     * @param {number} max - Valor máximo de la talla
     * @returns {string} - Descripción (tight, regular, loose)
     */
    getFitDescription(userValue, min, max) {
        // Calcular rango
        const range = max - min;
        
        // Definir umbrales
        const tightThreshold = min + (range * 0.2);
        const looseThreshold = max - (range * 0.2);
        
        // Determinar tipo de ajuste
        if (userValue < min) {
            // Por debajo del mínimo
            return 'very-tight';
        } else if (userValue <= tightThreshold) {
            // En el primer 20% del rango
            return 'tight';
        } else if (userValue >= looseThreshold) {
            // En el último 20% del rango
            return 'loose';
        } else if (userValue > max) {
            // Por encima del máximo
            return 'very-loose';
        } else {
            // En el centro del rango
            return 'regular';
        }
    }
    
    /**
     * Muestra la recomendación en la UI
     * @param {Object} recommendation - Resultado de la recomendación
     * @param {Object} clothing - Información de la prenda
     */
    displayRecommendation(recommendation, clothing) {
        if (!this.recommendationElement) return;
        
        // Asegurarse de que tenemos una recomendación
        if (!recommendation || !recommendation.recommendedSize) {
            this.showError("No se pudo calcular una recomendación de talla");
            return;
        }
        
        // Mostrar talla recomendada
        this.recommendationElement.innerHTML = `
            <div class="recommended-size-wrapper">
                <div class="recommended-size-label">Talla recomendada</div>
                <div class="recommended-size">${recommendation.recommendedSize}</div>
                <div class="fit-score">
                    <div class="fit-score-bar">
                        <div class="fit-score-fill" style="width: ${recommendation.fitScore}%"></div>
                    </div>
                    <div class="fit-score-label">${Math.round(recommendation.fitScore)}% de ajuste</div>
                </div>
            </div>
        `;
        
        // Mostrar tallas alternativas
        this.displayAlternatives(recommendation.alternatives);
        
        // Mostrar información de ajuste
        this.displayFitInfo(recommendation, clothing);
        
        // Mostrar explicación detallada
        this.displayFitExplanation(recommendation);
    }
    
    /**
     * Muestra tallas alternativas
     * @param {Array} alternatives - Tallas alternativas
     */
    displayAlternatives(alternatives) {
        if (!this.alternativesElement) return;
        
        if (!alternatives || alternatives.length === 0) {
            this.alternativesElement.innerHTML = '';
            return;
        }
        
        let html = '<div class="alternatives-label">Tallas alternativas</div>';
        html += '<div class="alternatives-container">';
        
        alternatives.forEach(alt => {
            html += `
                <div class="alternative-size">
                    <div class="alternative-size-label">${alt.size}</div>
                    <div class="alternative-score">${Math.round(alt.score)}%</div>
                </div>
            `;
        });
        
        html += '</div>';
        this.alternativesElement.innerHTML = html;
    }
    
    /**
     * Muestra información general de ajuste
     * @param {Object} recommendation - Resultado de la recomendación
     * @param {Object} clothing - Información de la prenda
     */
    displayFitInfo(recommendation, clothing) {
        if (!this.fitInfoElement) return;
        
        // Determinar tipo de ajuste general basado en los detalles
        const fitTypes = Object.values(recommendation.details).map(d => d.fit);
        
        // Contar cada tipo de ajuste
        const fitCounts = {
            'very-tight': 0,
            'tight': 0,
            'regular': 0,
            'loose': 0,
            'very-loose': 0
        };
        
        fitTypes.forEach(fit => {
            if (fitCounts[fit] !== undefined) {
                fitCounts[fit]++;
            }
        });
        
        // Determinar ajuste predominante
        let predominantFit = 'regular';
        let maxCount = 0;
        
        Object.entries(fitCounts).forEach(([fit, count]) => {
            if (count > maxCount) {
                maxCount = count;
                predominantFit = fit;
            }
        });
        
        // Generar descripción basada en el ajuste predominante
        let fitDescription = '';
        let fitClass = '';
        
        switch (predominantFit) {
            case 'very-tight':
                fitDescription = 'Esta talla podría resultarte muy ajustada';
                fitClass = 'very-tight';
                break;
            case 'tight':
                fitDescription = 'Esta talla te quedará ajustada';
                fitClass = 'tight';
                break;
            case 'regular':
                fitDescription = 'Esta talla te quedará bien';
                fitClass = 'regular';
                break;
            case 'loose':
                fitDescription = 'Esta talla te quedará holgada';
                fitClass = 'loose';
                break;
            case 'very-loose':
                fitDescription = 'Esta talla podría resultarte muy holgada';
                fitClass = 'very-loose';
                break;
        }
        
        // Generar HTML
        this.fitInfoElement.innerHTML = `
            <div class="fit-info-label">Cómo te quedará</div>
            <div class="fit-description ${fitClass}">
                <i class="fas fa-tshirt"></i>
                <span>${fitDescription}</span>
            </div>
        `;
    }
    
    /**
     * Muestra explicación detallada del ajuste
     * @param {Object} recommendation - Resultado de la recomendación
     */
    displayFitExplanation(recommendation) {
        if (!this.fitExplanationElement) return;
        
        if (!recommendation.details || Object.keys(recommendation.details).length === 0) {
            this.fitExplanationElement.innerHTML = '';
            return;
        }
        
        // Mapeo de nombres de medidas a etiquetas legibles
        const measurementLabels = {
            'chest': 'Pecho',
            'waist': 'Cintura',
            'hips': 'Cadera',
            'shoulder': 'Hombros',
            'inseam': 'Entrepierna'
        };
        
        // Generar tabla de explicación
        let html = '<div class="explanation-label">Detalles del ajuste</div>';
        html += '<table class="explanation-table">';
        html += `
            <tr>
                <th>Medida</th>
                <th>Tu medida</th>
                <th>Rango de talla</th>
                <th>Ajuste</th>
            </tr>
        `;
        
        Object.entries(recommendation.details).forEach(([measurement, detail]) => {
            const label = measurementLabels[measurement] || measurement;
            const fitClass = detail.fit;
            
            let fitText = '';
            switch (detail.fit) {
                case 'very-tight':
                    fitText = 'Muy ajustado';
                    break;
                case 'tight':
                    fitText = 'Ajustado';
                    break;
                case 'regular':
                    fitText = 'Regular';
                    break;
                case 'loose':
                    fitText = 'Holgado';
                    break;
                case 'very-loose':
                    fitText = 'Muy holgado';
                    break;
            }
            
            html += `
                <tr>
                    <td>${label}</td>
                    <td>${detail.user} cm</td>
                    <td>${detail.min}-${detail.max} cm</td>
                    <td class="${fitClass}">${fitText}</td>
                </tr>
            `;
        });
        
        html += '</table>';
        
        // Añadir consejo personalizado
        html += '<div class="fit-advice">';
        
        const fitScores = {
            'very-tight': Object.values(recommendation.details).filter(d => d.fit === 'very-tight').length,
            'tight': Object.values(recommendation.details).filter(d => d.fit === 'tight').length,
            'regular': Object.values(recommendation.details).filter(d => d.fit === 'regular').length,
            'loose': Object.values(recommendation.details).filter(d => d.fit === 'loose').length,
            'very-loose': Object.values(recommendation.details).filter(d => d.fit === 'very-loose').length
        };
        
        // Consejos específicos basados en el ajuste
        if (fitScores['very-tight'] > 0 || fitScores['tight'] > 1) {
            html += '<p><strong>Consejo:</strong> Si prefieres un ajuste más cómodo, considera una talla más grande.</p>';
        } else if (fitScores['very-loose'] > 0 || fitScores['loose'] > 1) {
            html += '<p><strong>Consejo:</strong> Si prefieres un ajuste más ceñido, considera una talla más pequeña.</p>';
        } else {
            html += '<p><strong>Consejo:</strong> Esta talla parece adecuada para tus medidas.</p>';
        }
        
        html += '</div>';
        
        this.fitExplanationElement.innerHTML = html;
    }
}

// Exportar la clase para uso en otros módulos
export default SizeRecommender;