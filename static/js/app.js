/**
 * app.js - Punto de entrada principal para la aplicación de prueba virtual
 * 
 * Este archivo maneja la inicialización de la aplicación, la navegación,
 * y la coordinación entre los diferentes módulos de la aplicación.
 */

// Esperar a que el DOM esté completamente cargado
document.addEventListener('DOMContentLoaded', () => {
    console.log('Virtual Fitting Room App Initialized');
    
    // Inicializar la aplicación
    App.init();
});

/**
 * Namespace principal de la aplicación
 */
const App = {
    // Configuración de la aplicación
    config: {
        apiEndpoint: '/api',
        debug: false,
        autoSave: true,
        defaultLanguage: 'es',
        supportedLanguages: ['es', 'en', 'fr'],
        maxImageSize: 5 * 1024 * 1024, // 5MB
        acceptedImageTypes: ['image/jpeg', 'image/png']
    },
    
    // Estado de la aplicación
    state: {
        currentUser: null,
        isAuthenticated: false,
        currentPage: null,
        measurements: null,
        selectedClothing: null,
        fittingResult: null,
        loading: false,
        error: null
    },
    
    // Referencias a elementos DOM importantes
    elements: {
        loadingOverlay: document.getElementById('loading-overlay'),
        errorMessage: document.getElementById('error-message'),
        navLinks: document.querySelectorAll('.nav-link'),
        userProfileButton: document.getElementById('user-profile-button'),
        languageSelector: document.getElementById('language-selector')
    },
    
    // Módulos de la aplicación
    modules: {
        bodyDetector: null,
        measurementUI: null,
        clothingFitter: null,
        sizeRecommender: null
    },
    
    /**
     * Inicializa la aplicación
     */
    init() {
        this.loadConfiguration();
        this.setupEventListeners();
        this.setupNavigation();
        this.initCurrentPage();
        this.checkAuthentication();
        
        // Inicializar módulos según la página actual
        this.initModules();
    },
    
    /**
     * Carga la configuración desde el servidor
     */
    async loadConfiguration() {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/config`);
            if (response.ok) {
                const serverConfig = await response.json();
                // Fusionar configuración del servidor con la configuración predeterminada
                this.config = { ...this.config, ...serverConfig };
                
                if (this.config.debug) {
                    console.log('Configuration loaded:', this.config);
                }
            }
        } catch (error) {
            console.error('Error loading configuration:', error);
        }
    },
    
    /**
     * Configura los event listeners para elementos globales de la interfaz
     */
    setupEventListeners() {
        // Manejador para el menú de navegación móvil
        const navToggle = document.querySelector('.nav-toggle');
        if (navToggle) {
            navToggle.addEventListener('click', this.toggleMobileNav.bind(this));
        }
        
        // Manejador para el selector de idiomas
        if (this.elements.languageSelector) {
            this.elements.languageSelector.addEventListener('change', (e) => {
                this.changeLanguage(e.target.value);
            });
        }
        
        // Manejador para el botón de perfil de usuario
        if (this.elements.userProfileButton) {
            this.elements.userProfileButton.addEventListener('click', this.showUserProfile.bind(this));
        }
        
        // Manejador global para errores de fetch
        window.addEventListener('unhandledrejection', (event) => {
            if (event.reason instanceof Error) {
                this.handleError(event.reason);
            }
        });
    },
    
    /**
     * Configura la navegación entre páginas
     */
    setupNavigation() {
        // Añadir event listeners a los enlaces de navegación
        this.elements.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetPage = link.getAttribute('data-page');
                if (targetPage) {
                    this.navigateTo(targetPage);
                }
            });
        });
        
        // Manejar navegación del historial del navegador
        window.addEventListener('popstate', (e) => {
            if (e.state && e.state.page) {
                this.loadPage(e.state.page, false);
            }
        });
    },
    
    /**
     * Inicializa la página actual según la URL
     */
    initCurrentPage() {
        // Determinar la página actual a partir de la URL
        const path = window.location.pathname;
        let page = 'home';
        
        if (path.includes('/measurement')) {
            page = 'measurement';
        } else if (path.includes('/fitting')) {
            page = 'fitting';
        } else if (path.includes('/results')) {
            page = 'results';
        }
        
        this.state.currentPage = page;
        this.updateActiveNavLink();
    },
    
    /**
     * Verifica si el usuario está autenticado
     */
    async checkAuthentication() {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/auth/check`, {
                credentials: 'include'
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.authenticated) {
                    this.state.isAuthenticated = true;
                    this.state.currentUser = data.user;
                    this.updateUserInterface();
                } else {
                    // Si algunas páginas requieren autenticación, redirigir al login
                    const requiresAuth = ['profile', 'measurement', 'results'];
                    if (requiresAuth.includes(this.state.currentPage)) {
                        this.navigateTo('login');
                    }
                }
            }
        } catch (error) {
            console.error('Error checking authentication:', error);
        }
    },
    
    /**
     * Inicializa los módulos necesarios según la página actual
     */
    initModules() {
        // Limpiar módulos existentes
        Object.keys(this.modules).forEach(key => {
            this.modules[key] = null;
        });
        
        // Inicializar módulos según la página actual
        switch (this.state.currentPage) {
            case 'measurement':
                this.initMeasurementPage();
                break;
            case 'fitting':
                this.initFittingPage();
                break;
            case 'results':
                this.initResultsPage();
                break;
            default:
                // Módulos comunes para todas las páginas
                break;
        }
    },
    
    /**
     * Inicializa la página de medición
     */
    async initMeasurementPage() {
        this.showLoading();
        
        try {
            // Importar dinámicamente los módulos necesarios
            const BodyDetector = (await import('./bodyDetector.js')).default;
            const MeasurementUI = (await import('./measurementUI.js')).default;
            
            // Inicializar módulos con la configuración necesaria
            this.modules.bodyDetector = new BodyDetector({
                modelPath: '/models/pose/mediapipe_pose_landmarker.task',
                canvas: document.getElementById('detection-canvas'),
                video: document.getElementById('user-video')
            });
            
            this.modules.measurementUI = new MeasurementUI({
                resultsContainer: document.getElementById('measurement-results'),
                captureButton: document.getElementById('capture-button'),
                uploadButton: document.getElementById('upload-button'),
                instructionsContainer: document.getElementById('measurement-instructions'),
                bodyDetector: this.modules.bodyDetector
            });
            
            // Inicializar la detección corporal
            await this.modules.bodyDetector.initialize();
            
            // Configurar la interfaz de medición
            this.modules.measurementUI.setup();
            
            // Cargar medidas anteriores si existen
            await this.loadPreviousMeasurements();
        } catch (error) {
            this.handleError(error);
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Inicializa la página de prueba de prendas
     */
    async initFittingPage() {
        this.showLoading();
        
        try {
            // Importar dinámicamente los módulos necesarios
            const ClothingFitter = (await import('./clothingFitter.js')).default;
            const SizeRecommender = (await import('./sizeRecommender.js')).default;
            
            // Verificar si tenemos medidas, si no, redirigir a la página de medición
            if (!this.state.measurements) {
                const measurements = await this.loadPreviousMeasurements();
                if (!measurements) {
                    this.navigateTo('measurement');
                    return;
                }
            }
            
            // Inicializar módulos
            this.modules.clothingFitter = new ClothingFitter({
                container: document.getElementById('fitting-preview'),
                measurements: this.state.measurements
            });
            
            this.modules.sizeRecommender = new SizeRecommender({
                container: document.getElementById('size-recommendation'),
                measurements: this.state.measurements
            });
            
            // Cargar las prendas disponibles
            await this.loadAvailableClothing();
            
            // Configurar los event listeners para la selección de prendas
            this.setupClothingSelection();
        } catch (error) {
            this.handleError(error);
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Inicializa la página de resultados
     */
    async initResultsPage() {
        this.showLoading();
        
        try {
            // Verificar si tenemos resultados de ajuste, si no, redirigir a la página de ajuste
            if (!this.state.fittingResult && !this.state.selectedClothing) {
                this.navigateTo('fitting');
                return;
            }
            
            // Cargar los resultados
            await this.loadFittingResults();
            
            // Mostrar los resultados en la página
            this.renderResults();
        } catch (error) {
            this.handleError(error);
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Carga las medidas anteriores del usuario
     */
    async loadPreviousMeasurements() {
        if (!this.state.isAuthenticated) return null;
        
        try {
            const response = await fetch(`${this.config.apiEndpoint}/measurements/user/${this.state.currentUser.id}`);
            if (response.ok) {
                const measurements = await response.json();
                this.state.measurements = measurements;
                return measurements;
            }
        } catch (error) {
            console.error('Error loading measurements:', error);
        }
        
        return null;
    },
    
    /**
     * Carga las prendas disponibles
     */
    async loadAvailableClothing() {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/clothing`);
            if (response.ok) {
                const clothing = await response.json();
                
                // Mostrar las prendas en la interfaz
                const container = document.getElementById('clothing-grid');
                if (container) {
                    container.innerHTML = '';
                    
                    clothing.forEach(item => {
                        const itemElement = document.createElement('div');
                        itemElement.className = 'clothing-item';
                        itemElement.setAttribute('data-id', item.id);
                        
                        itemElement.innerHTML = `
                            <img src="${item.thumbnail}" alt="${item.name}">
                            <div class="clothing-info">
                                <h3>${item.name}</h3>
                                <p>${item.brand}</p>
                            </div>
                        `;
                        
                        itemElement.addEventListener('click', () => {
                            this.selectClothing(item);
                        });
                        
                        container.appendChild(itemElement);
                    });
                }
                
                return clothing;
            }
        } catch (error) {
            console.error('Error loading clothing:', error);
        }
        
        return [];
    },
    
    /**
     * Configura la selección de prendas
     */
    setupClothingSelection() {
        // Configurar event listeners para los filtros
        const filterButtons = document.querySelectorAll('.filter-button');
        filterButtons.forEach(button => {
            button.addEventListener('click', () => {
                const category = button.getAttribute('data-category');
                this.filterClothing(category);
                
                // Actualizar UI para mostrar el filtro activo
                filterButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
            });
        });
    },
    
    /**
     * Filtra las prendas por categoría
     */
    filterClothing(category) {
        const items = document.querySelectorAll('.clothing-item');
        
        if (category === 'all') {
            items.forEach(item => item.style.display = 'block');
        } else {
            items.forEach(item => {
                const itemCategory = item.getAttribute('data-category');
                item.style.display = itemCategory === category ? 'block' : 'none';
            });
        }
    },
    
    /**
     * Selecciona una prenda para probar
     */
    async selectClothing(clothing) {
        this.state.selectedClothing = clothing;
        
        // Actualizar UI para mostrar la prenda seleccionada
        const selectedItems = document.querySelectorAll('.clothing-item.selected');
        selectedItems.forEach(item => item.classList.remove('selected'));
        
        const itemElement = document.querySelector(`.clothing-item[data-id="${clothing.id}"]`);
        if (itemElement) {
            itemElement.classList.add('selected');
        }
        
        // Mostrar la prenda en el visualizador
        if (this.modules.clothingFitter) {
            await this.modules.clothingFitter.fitClothing(clothing);
        }
        
        // Mostrar recomendación de talla
        if (this.modules.sizeRecommender) {
            await this.modules.sizeRecommender.recommendSize(clothing);
        }
    },
    
    /**
     * Carga los resultados de ajuste
     */
    async loadFittingResults() {
        if (!this.state.selectedClothing) return;
        
        try {
            const response = await fetch(`${this.config.apiEndpoint}/fitting/results`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    userId: this.state.currentUser?.id,
                    clothingId: this.state.selectedClothing.id,
                    measurements: this.state.measurements
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                this.state.fittingResult = result;
                return result;
            }
        } catch (error) {
            console.error('Error loading fitting results:', error);
        }
        
        return null;
    },
    
    /**
     * Muestra los resultados en la página
     */
    renderResults() {
        if (!this.state.fittingResult) return;
        
        // Actualizar la imagen de vista previa
        const previewImage = document.getElementById('result-preview-image');
        if (previewImage) {
            previewImage.src = this.state.fittingResult.previewImage;
        }
        
        // Actualizar información de la prenda
        const clothingInfo = document.getElementById('clothing-info');
        if (clothingInfo && this.state.selectedClothing) {
            clothingInfo.innerHTML = `
                <h2>${this.state.selectedClothing.name}</h2>
                <p class="brand">${this.state.selectedClothing.brand}</p>
                <p class="description">${this.state.selectedClothing.description}</p>
                <div class="price">${this.state.selectedClothing.price}</div>
            `;
        }
        
        // Mostrar recomendación de talla
        const sizeRecommendation = document.getElementById('size-recommendation');
        if (sizeRecommendation) {
            sizeRecommendation.innerHTML = `
                <div class="recommendation-header">Talla recomendada</div>
                <div class="recommended-size">${this.state.fittingResult.recommendedSize}</div>
                <div class="fit-description">${this.state.fittingResult.fitDescription}</div>
            `;
        }
        
        // Actualizar tabla de medidas
        const measurementsTable = document.getElementById('measurements-table');
        if (measurementsTable && this.state.fittingResult.measurements) {
            let tableHtml = `
                <tr>
                    <th>Medida</th>
                    <th>Tus medidas</th>
                    <th>Medidas de la prenda</th>
                    <th>Diferencia</th>
                </tr>
            `;
            
            Object.entries(this.state.fittingResult.measurements).forEach(([key, value]) => {
                tableHtml += `
                    <tr>
                        <td>${this.translateMeasurement(key)}</td>
                        <td>${value.user} cm</td>
                        <td>${value.garment} cm</td>
                        <td class="${value.difference > 0 ? 'positive' : value.difference < 0 ? 'negative' : 'neutral'}">
                            ${value.difference > 0 ? '+' : ''}${value.difference} cm
                        </td>
                    </tr>
                `;
            });
            
            measurementsTable.innerHTML = tableHtml;
        }
    },
    
    /**
     * Traduce las claves de medición a nombres legibles
     */
    translateMeasurement(key) {
        const translations = {
            'chest': 'Pecho',
            'waist': 'Cintura',
            'hips': 'Cadera',
            'shoulder': 'Hombros',
            'arm_length': 'Longitud de brazo',
            'leg_length': 'Longitud de pierna',
            'neck': 'Cuello'
        };
        
        return translations[key] || key;
    },
    
    /**
     * Navega a una página específica
     */
    navigateTo(page, addToHistory = true) {
        // Construir la URL para la página
        let url;
        
        switch (page) {
            case 'home':
                url = '/';
                break;
            case 'login':
                url = '/login';
                break;
            default:
                url = `/${page}`;
                break;
        }
        
        // Añadir a la historia del navegador si es necesario
        if (addToHistory) {
            window.history.pushState({ page }, `Virtual Fitting - ${page}`, url);
        }
        
        // Cargar la página
        this.loadPage(page);
    },
    
    /**
     * Carga una página específica
     */
    async loadPage(page, updateModules = true) {
        this.showLoading();
        
        try {
            // Actualizar el estado de la página actual
            this.state.currentPage = page;
            
            // Actualizar el enlace de navegación activo
            this.updateActiveNavLink();
            
            // Si es una SPA, cargar el contenido dinámicamente
            const contentContainer = document.getElementById('page-content');
            if (contentContainer) {
                const response = await fetch(`/templates/${page}.html`);
                if (response.ok) {
                    const html = await response.text();
                    contentContainer.innerHTML = html;
                    
                    // Inicializar los módulos para la nueva página
                    if (updateModules) {
                        this.initModules();
                    }
                }
            } else {
                // Si no es una SPA, redirigir a la nueva página
                window.location.href = `/${page}.html`;
            }
        } catch (error) {
            console.error('Error loading page:', error);
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Actualiza el enlace de navegación activo
     */
    updateActiveNavLink() {
        // Quitar la clase activa de todos los enlaces
        this.elements.navLinks.forEach(link => {
            link.classList.remove('active');
        });
        
        // Añadir la clase activa al enlace correspondiente a la página actual
        const activeLink = document.querySelector(`.nav-link[data-page="${this.state.currentPage}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    },
    
    /**
     * Alterna la visibilidad del menú de navegación móvil
     */
    toggleMobileNav() {
        const nav = document.querySelector('.main-nav');
        const overlay = document.querySelector('.nav-overlay');
        
        if (nav) {
            nav.classList.toggle('open');
        }
        
        if (overlay) {
            overlay.classList.toggle('open');
        }
    },
    
    /**
     * Cambia el idioma de la aplicación
     */
    changeLanguage(language) {
        if (this.config.supportedLanguages.includes(language)) {
            this.config.defaultLanguage = language;
            
            // Guardar preferencia del usuario
            localStorage.setItem('preferredLanguage', language);
            
            // Recargar la página para aplicar el cambio de idioma
            window.location.reload();
        }
    },
    
    /**
     * Muestra el perfil del usuario
     */
    showUserProfile() {
        if (this.state.isAuthenticated) {
            this.navigateTo('profile');
        } else {
            this.navigateTo('login');
        }
    },
    
    /**
     * Actualiza la interfaz de usuario según el estado de autenticación
     */
    updateUserInterface() {
        // Actualizar elementos que dependen del estado de autenticación
        const authElements = document.querySelectorAll('[data-auth-required]');
        const nonAuthElements = document.querySelectorAll('[data-auth-hidden]');
        
        authElements.forEach(el => {
            el.style.display = this.state.isAuthenticated ? 'block' : 'none';
        });
        
        nonAuthElements.forEach(el => {
            el.style.display = this.state.isAuthenticated ? 'none' : 'block';
        });
        
        // Actualizar nombre de usuario si está disponible
        const usernameElements = document.querySelectorAll('.username');
        usernameElements.forEach(el => {
            if (this.state.currentUser) {
                el.textContent = this.state.currentUser.name;
            }
        });
    },
    
    /**
     * Muestra el indicador de carga
     */
    showLoading() {
        this.state.loading = true;
        
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.style.display = 'flex';
        }
    },
    
    /**
     * Oculta el indicador de carga
     */
    hideLoading() {
        this.state.loading = false;
        
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.style.display = 'none';
        }
    },
    
    /**
     * Maneja errores en la aplicación
     */
    handleError(error) {
        console.error('Application error:', error);
        
        this.state.error = error.message || 'Se ha producido un error desconocido';
        
        // Mostrar mensaje de error en la interfaz
        if (this.elements.errorMessage) {
            this.elements.errorMessage.textContent = this.state.error;
            this.elements.errorMessage.style.display = 'block';
            
            // Ocultar automáticamente después de 5 segundos
            setTimeout(() => {
                this.elements.errorMessage.style.display = 'none';
            }, 5000);
        }
    }
};