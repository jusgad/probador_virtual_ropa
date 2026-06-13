"use client";

import React from "react";
import { motion } from "framer-motion";
import { Sparkles, ArrowRight, Cpu, Zap, ChevronRight, BarChart3, Database } from "lucide-react";
import FittingSimulator from "../components/FittingSimulator";

export default function Home() {
  return (
    <div className="relative min-h-screen bg-[#0B0F19] text-gray-100 overflow-hidden">
      
      {/* 1. Fondo de Nebulosas de Neón (Decorativos) */}
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-violet-600/10 blur-[120px] pointer-events-none animate-pulse-glow-violet" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-teal-500/10 blur-[120px] pointer-events-none animate-pulse-glow-teal" />
      
      {/* Patrón de Rejilla Tecnológica en el fondo */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.01)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.01)_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />

      {/* 2. Navbar Flotante Glassmorphic */}
      <header className="sticky top-4 z-50 max-w-7xl mx-auto px-4 w-full">
        <motion.nav 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="bg-slate-900/50 backdrop-blur-md border border-white/10 rounded-2xl px-6 py-4 flex items-center justify-between shadow-xl"
        >
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-600 to-teal-400 flex items-center justify-center font-bold text-white shadow-md shadow-violet-500/20">
              FV
            </div>
            <span className="text-xl font-black tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-gray-200 to-gray-400">
              FitVibe
            </span>
          </div>

          {/* Enlaces de Navegación */}
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-gray-400">
            <a href="#simulator" className="hover:text-white transition-colors">Simulador</a>
            <a href="#features" className="hover:text-white transition-colors">Características</a>
            <a href="#architecture" className="hover:text-white transition-colors">Arquitectura</a>
          </div>

          {/* Botón de Acción */}
          <div className="flex items-center gap-4">
            <a 
              href="#simulator"
              className="relative group overflow-hidden px-4.5 py-2 rounded-xl bg-white/5 border border-white/10 text-xs font-bold uppercase tracking-wider text-white hover:bg-white/10 transition-all duration-300"
            >
              <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-violet-600/20 to-teal-400/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              Probar Demo
            </a>
          </div>
        </motion.nav>
      </header>

      {/* 3. Hero Section */}
      <main className="max-w-7xl mx-auto px-4 pt-16 pb-24 relative z-10 flex flex-col items-center">
        
        {/* Badge superior */}
        <motion.div 
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-violet-500/10 border border-violet-500/30 text-xs font-bold text-violet-400 mb-6 uppercase tracking-wider"
        >
          <Sparkles className="w-3.5 h-3.5" />
          Ventas & Demo Interactiva
        </motion.div>

        {/* Gran Título Hero */}
        <div className="text-center max-w-4xl">
          <motion.h1 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-4xl sm:text-6xl md:text-7xl font-extrabold tracking-tight leading-none mb-6"
          >
            Vístete en el Futuro: <br className="hidden sm:inline" />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-400 via-fuchsia-400 to-teal-300">
              Probador Virtual con IA
            </span>
          </motion.h1>

          <motion.p 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-base sm:text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed mb-10"
          >
            Escanea tu cuerpo, calcula tus medidas con precisión milimétrica y pruébate prendas en tiempo real de tus marcas favoritas con nuestro motor inteligente de tolerancia.
          </motion.p>
        </div>

        {/* CTAs del Hero */}
        <motion.div 
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="flex flex-col sm:flex-row gap-4 mb-20 z-10"
        >
          <a
            href="#simulator"
            className="px-8 py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 rounded-2xl font-bold text-sm uppercase tracking-wider text-white shadow-xl shadow-violet-600/20 hover:shadow-violet-600/30 hover:scale-[1.02] active:scale-[0.98] transition-all duration-300 flex items-center justify-center gap-2"
          >
            Iniciar Probador Virtual
            <ArrowRight className="w-4 h-4" />
          </a>
          <a
            href="#features"
            className="px-8 py-4 bg-slate-900/40 backdrop-blur-sm border border-white/10 rounded-2xl font-bold text-sm uppercase tracking-wider text-white hover:bg-white/5 transition-all duration-300 flex items-center justify-center"
          >
            Ver Características
          </a>
        </motion.div>

        {/* Tarjetas de Estadísticas/Indicadores del Hero */}
        <motion.div 
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.4 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full max-w-5xl mb-24"
        >
          <div className="bg-slate-900/30 backdrop-blur-md border border-white/5 rounded-2xl p-5 text-center hover:border-white/10 transition-colors">
            <span className="block text-2xl sm:text-3xl font-black text-violet-400 mb-1">98.7%</span>
            <span className="text-[11px] uppercase tracking-wider font-semibold text-gray-500">Precisión Biométrica</span>
          </div>
          <div className="bg-slate-900/30 backdrop-blur-md border border-white/5 rounded-2xl p-5 text-center hover:border-white/10 transition-colors">
            <span className="block text-2xl sm:text-3xl font-black text-teal-400 mb-1">&lt; 0.5s</span>
            <span className="text-[11px] uppercase tracking-wider font-semibold text-gray-500">Latencia de Pose</span>
          </div>
          <div className="bg-slate-900/30 backdrop-blur-md border border-white/5 rounded-2xl p-5 text-center hover:border-white/10 transition-colors">
            <span className="block text-2xl sm:text-3xl font-black text-violet-400 mb-1">5+</span>
            <span className="text-[11px] uppercase tracking-wider font-semibold text-gray-500">Marcas Integradas</span>
          </div>
          <div className="bg-slate-900/30 backdrop-blur-md border border-white/5 rounded-2xl p-5 text-center hover:border-white/10 transition-colors">
            <span className="block text-2xl sm:text-3xl font-black text-teal-400 mb-1">100%</span>
            <span className="text-[11px] uppercase tracking-wider font-semibold text-gray-500">Thread-Safe SQLite</span>
          </div>
        </motion.div>

        {/* 4. Sección del Simulador Interactivo */}
        <section id="simulator" className="w-full max-w-6xl mb-28 scroll-mt-24">
          <div className="text-center mb-10">
            <span className="text-xs font-bold uppercase tracking-widest text-violet-400 block mb-2">Simulación Interactiva</span>
            <h2 className="text-2xl sm:text-4xl font-extrabold text-white">Prueba en Tiempo Real con Estimación de Pose</h2>
          </div>
          
          <FittingSimulator />
        </section>

        {/* 5. Sección de Características Grid Asimétrico */}
        <section id="features" className="w-full max-w-6xl mb-28 scroll-mt-24">
          <div className="text-center mb-14">
            <span className="text-xs font-bold uppercase tracking-widest text-teal-400 block mb-2">Tecnología de Punta</span>
            <h2 className="text-2xl sm:text-4xl font-extrabold text-white">¿Cómo Funciona FitVibe?</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            {/* Card 1: Pose Estimation (7 cols) */}
            <div className="md:col-span-7 bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-8 relative overflow-hidden group hover:border-violet-500/30 transition-all duration-300 flex flex-col justify-between min-h-[300px]">
              <div className="absolute top-0 right-0 w-32 h-32 bg-violet-600/10 rounded-full blur-2xl group-hover:bg-violet-600/20 transition-all" />
              <div>
                <div className="w-12 h-12 rounded-2xl bg-violet-500/20 border border-violet-500/40 flex items-center justify-center text-violet-400 mb-6">
                  <Cpu className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Estimación de Pose de 33 Puntos</h3>
                <p className="text-sm text-gray-400 leading-relaxed">
                  Utilizamos la tecnología moderna de MediaPipe Tasks para detectar puntos clave anatómicos en milisegundos. Captura de hombros, cintura, caderas y extremidades para calcular una réplica digital de tu silueta.
                </p>
              </div>
              <span className="text-xs font-bold text-violet-400 mt-6 flex items-center gap-1">
                Ver Documentación de MediaPipe <ChevronRight className="w-4 h-4" />
              </span>
            </div>

            {/* Card 2: Tolerancia (5 cols) */}
            <div className="md:col-span-5 bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-8 relative overflow-hidden group hover:border-teal-500/30 transition-all duration-300 flex flex-col justify-between min-h-[300px]">
              <div className="absolute top-0 right-0 w-32 h-32 bg-teal-500/10 rounded-full blur-2xl group-hover:bg-teal-500/20 transition-all" />
              <div>
                <div className="w-12 h-12 rounded-2xl bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-400 mb-6">
                  <Zap className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Algoritmo de Holgura</h3>
                <p className="text-sm text-gray-400 leading-relaxed">
                  Las prendas no deben quedar pegadas como papel. Nuestro algoritmo calcula la tolerancia necesaria según la tela y el estilo (Slim, Regular, Oversized) para ofrecerte la recomendación de talla ideal.
                </p>
              </div>
              <span className="text-xs font-bold text-teal-400 mt-6 flex items-center gap-1">
                Ver Algoritmo de Tallas <ChevronRight className="w-4 h-4" />
              </span>
            </div>

            {/* Card 3: SQLite de Alta Seguridad (5 cols) */}
            <div className="md:col-span-5 bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-8 relative overflow-hidden group hover:border-violet-500/30 transition-all duration-300 flex flex-col justify-between min-h-[300px]">
              <div className="absolute top-0 right-0 w-32 h-32 bg-violet-600/10 rounded-full blur-2xl group-hover:bg-violet-600/20 transition-all" />
              <div>
                <div className="w-12 h-12 rounded-2xl bg-violet-500/20 border border-violet-500/40 flex items-center justify-center text-violet-400 mb-6">
                  <Database className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Base de Datos Concurrente</h3>
                <p className="text-sm text-gray-400 leading-relaxed">
                  Motor SQLite optimizado con pooling de conexiones local por hilo (`threading.local`) para garantizar transacciones thread-safe en entornos web multihilo y hashes PBKDF2 de alta seguridad para contraseñas.
                </p>
              </div>
              <span className="text-xs font-bold text-violet-400 mt-6 flex items-center gap-1">
                Ver Configuración SQLite <ChevronRight className="w-4 h-4" />
              </span>
            </div>

            {/* Card 4: Compatibilidad SPA Completa (7 cols) */}
            <div className="md:col-span-7 bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-8 relative overflow-hidden group hover:border-teal-500/30 transition-all duration-300 flex flex-col justify-between min-h-[300px]">
              <div className="absolute top-0 right-0 w-32 h-32 bg-teal-500/10 rounded-full blur-2xl group-hover:bg-teal-500/20 transition-all" />
              <div>
                <div className="w-12 h-12 rounded-2xl bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-400 mb-6">
                  <BarChart3 className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">API SPA Unificada</h3>
                <p className="text-sm text-gray-400 leading-relaxed">
                  Perfectamente adaptada para arquitecturas Single Page Application con enrutamiento dinámico en cliente y backend desacoplado, exponiendo endpoints rápidos de control de sesión, charts de tallas y renderizado de resultados.
                </p>
              </div>
              <span className="text-xs font-bold text-teal-400 mt-6 flex items-center gap-1">
                Ver Documentación de API <ChevronRight className="w-4 h-4" />
              </span>
            </div>
          </div>
        </section>

        {/* 6. Footer */}
        <footer className="w-full max-w-6xl border-t border-white/5 pt-12 flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-gray-500">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-violet-600 flex items-center justify-center font-bold text-[10px] text-white">
              FV
            </div>
            <span>© 2026 FitVibe. Todos los derechos reservados.</span>
          </div>
          <div className="flex gap-6">
            <a href="#simulator" className="hover:text-white transition-colors">Terminos de Servicio</a>
            <a href="#simulator" className="hover:text-white transition-colors">Privacidad</a>
            <a href="#simulator" className="hover:text-white transition-colors">Demo Admin</a>
          </div>
        </footer>

      </main>
    </div>
  );
}
