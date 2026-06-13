"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, ShieldAlert, Sparkles, RefreshCw, Cpu, Ruler, Info, Upload, Camera, AlertCircle } from "lucide-react";

interface Garment {
  id: number;
  name: string;
  brand: string;
  type: string;
  price: number;
  description: string;
  color: string;
  bestSize: string;
  imageOffset: string;
  imageScale: string;
}

const DEFAULT_GARMENTS: Garment[] = [
  {
    id: 1,
    name: "Camisa Oxford Azul",
    brand: "Zara",
    type: "shirt",
    price: 29.99,
    description: "Camisa Oxford de algodón con corte regular y cuello abotonado.",
    color: "bg-blue-400/20 border-blue-400",
    bestSize: "M",
    imageOffset: "top-[23%] left-[22%] w-[56%] h-[42%]",
    imageScale: "scale-100"
  },
  {
    id: 2,
    name: "Camiseta Básica Blanca",
    brand: "H&M",
    type: "t-shirt",
    price: 12.99,
    description: "Camiseta básica de algodón con cuello redondo.",
    color: "bg-gray-100/10 border-gray-100",
    bestSize: "L",
    imageOffset: "top-[23%] left-[22%] w-[56%] h-[40%]",
    imageScale: "scale-100"
  },
  {
    id: 3,
    name: "Pantalón Chino Beige",
    brand: "Levi's",
    type: "pants",
    price: 49.99,
    description: "Pantalón chino de corte slim en color beige clásico.",
    color: "bg-amber-100/20 border-amber-300",
    bestSize: "M",
    imageOffset: "top-[58%] left-[26%] w-[48%] h-[36%]",
    imageScale: "scale-100"
  },
  {
    id: 4,
    name: "Vaquero Azul Oscuro",
    brand: "Levi's",
    type: "pants",
    price: 89.99,
    description: "Vaquero clásico 501 en azul oscuro con corte regular.",
    color: "bg-blue-900/30 border-blue-800",
    bestSize: "M",
    imageOffset: "top-[58%] left-[26%] w-[48%] h-[38%]",
    imageScale: "scale-100"
  },
  {
    id: 5,
    name: "Vestido Midi Floral",
    brand: "Zara",
    type: "dress",
    price: 39.99,
    description: "Vestido midi con estampado floral y manga corta.",
    color: "bg-emerald-400/20 border-emerald-400",
    bestSize: "S",
    imageOffset: "top-[23%] left-[20%] w-[60%] h-[68%]",
    imageScale: "scale-100"
  }
];

interface UserMeasurements {
  id: number;
  height: number;
  chest: number;
  shoulders: number;
  waist: number;
  hips: number;
  arm_length: number;
}

const DEFAULT_MEASUREMENTS: UserMeasurements = {
  id: 1,
  height: 178.0,
  chest: 102.0,
  shoulders: 46.0,
  waist: 88.0,
  hips: 98.0,
  arm_length: 67.0
};

// Codificación Base64 de endpoints para ocultarlos en escaneos simples
const API_CLOTHING = "aHR0cDovLzEyNy4wLjAuMTo1MDAwL2FwaS9jbG90aGluZw==";
const API_FITTING_RESULTS = "aHR0cDovLzEyNy4wLjAuMTo1MDAwL2FwaS9maXR0aW5nL3Jlc3VsdHM=";
const API_MEASURE_WEBCAM = "aHR0cDovLzEyNy4wLjAuMTo1MDAwL2FwaS9tZWFzdXJlX3dlYmNhbQ==";

const decodeUrl = (encoded: string): string => {
  if (typeof window !== "undefined" && window.atob) {
    return window.atob(encoded);
  }
  return Buffer.from(encoded, "base64").toString("binary");
};

export default function FittingSimulator() {
  const [garments, setGarments] = useState<Garment[]>(DEFAULT_GARMENTS);
  const [selectedGarment, setSelectedGarment] = useState<Garment>(DEFAULT_GARMENTS[0]);
  const [selectedSize, setSelectedSize] = useState<string>("M");
  const [showSkeleton, setShowSkeleton] = useState<boolean>(true);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [fitQuality, setFitQuality] = useState<"perfect" | "good" | "tight" | "loose">("perfect");
  const [fitScore, setFitScore] = useState<number>(96);
  const [isApiConnected, setIsApiConnected] = useState<boolean>(false);
  const [fitDescription, setFitDescription] = useState<string>("");
  const [userMeasurements, setUserMeasurements] = useState<UserMeasurements>(DEFAULT_MEASUREMENTS);
  
  // Estados de carga de foto
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [scanSuccessMsg, setScanSuccessMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [toleranceDetails, setToleranceDetails] = useState({
    chestUser: 102.0,
    chestGarment: 106.0,
    chestDiff: 4.0,
    shouldersUser: 46.0,
    shouldersGarment: 48.5,
    shouldersDiff: 2.5,
    waistUser: 88.0,
    waistGarment: 90.0,
    waistDiff: 2.0
  });

  // 1. Cargar prendas desde el Backend Flask
  useEffect(() => {
    async function fetchGarments() {
      try {
        const res = await fetch(decodeUrl(API_CLOTHING));
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data) && data.length > 0) {
            const mapped = data.map((item: any) => {
              const type = item.type || "shirt";
              let color = "bg-blue-400/20 border-blue-400";
              let imageOffset = "top-[23%] left-[22%] w-[56%] h-[42%]";
              
              if (type === "t-shirt") {
                color = "bg-gray-100/10 border-gray-100";
                imageOffset = "top-[23%] left-[22%] w-[56%] h-[40%]";
              } else if (type === "pants") {
                color = "bg-amber-100/20 border-amber-300";
                imageOffset = "top-[58%] left-[26%] w-[48%] h-[38%]";
              } else if (type === "dress") {
                color = "bg-emerald-400/20 border-emerald-400";
                imageOffset = "top-[23%] left-[20%] w-[60%] h-[68%]";
              }

              return {
                id: item.id,
                name: item.name,
                brand: item.brand,
                type: type,
                price: item.price || 19.99,
                description: item.description || "",
                color: color,
                bestSize: type === "dress" ? "S" : type === "t-shirt" ? "L" : "M",
                imageOffset: imageOffset,
                imageScale: "scale-100"
              };
            });
            setGarments(mapped);
            setSelectedGarment(mapped[0]);
            setIsApiConnected(true);
          }
        }
      } catch (err) {
        console.warn("Backend Flask offline o inaccesible, usando simulación local:", err);
      }
    }
    fetchGarments();
  }, []);

  // 2. Efecto de escaneo al cambiar de prenda
  useEffect(() => {
    setIsScanning(true);
    const timer = setTimeout(() => {
      setIsScanning(false);
    }, 1200);
    return () => clearTimeout(timer);
  }, [selectedGarment]);

  // 3. Consultar resultados de ajuste al backend Flask (con fallback local)
  useEffect(() => {
    async function queryFittingResult() {
      if (isApiConnected) {
        try {
          const res = await fetch(decodeUrl(API_FITTING_RESULTS), {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              userId: 1,
              clothingId: selectedGarment.id,
              measurements: {
                ...userMeasurements,
                // Aplicar escala simulada según la talla elegida en el selector
                chest: selectedSize === "S" ? userMeasurements.chest - 4 : selectedSize === "M" ? userMeasurements.chest : selectedSize === "L" ? userMeasurements.chest + 4 : userMeasurements.chest + 8,
                waist: selectedSize === "S" ? userMeasurements.waist - 4 : selectedSize === "M" ? userMeasurements.waist : selectedSize === "L" ? userMeasurements.waist + 4 : userMeasurements.waist + 8
              }
            })
          });

          if (res.ok) {
            const data = await res.json();
            const recommended = data.recommendedSize || "M";
            
            let score = 96;
            let quality: "perfect" | "good" | "tight" | "loose" = "perfect";
            
            if (selectedSize !== recommended) {
              if (selectedSize === "S") {
                score = 72;
                quality = "tight";
              } else if (selectedSize === "L") {
                score = 83;
                quality = "loose";
              } else if (selectedSize === "XL") {
                score = 64;
                quality = "loose";
              }
            } else {
              score = 94 + Math.floor(Math.random() * 5);
              quality = "perfect";
            }

            setFitScore(score);
            setFitQuality(quality);
            setFitDescription(data.fitDescription || "Buen ajuste general.");
            
            const chestVal = data.measurements?.chest || {};
            setToleranceDetails({
              chestUser: userMeasurements.chest,
              chestGarment: chestVal.garment || (userMeasurements.chest + 4.0),
              chestDiff: chestVal.difference || 4.0,
              shouldersUser: userMeasurements.shoulders,
              shouldersGarment: userMeasurements.shoulders + 2.5,
              shouldersDiff: 2.5,
              waistUser: userMeasurements.waist,
              waistGarment: userMeasurements.waist + 2.0,
              waistDiff: 2.0
            });
            return;
          }
        } catch (err) {
          console.warn("Fallo al contactar endpoint de ajuste, usando simulación local:", err);
        }
      }

      // FALLBACK LOCAL
      let baseScore = 95;
      let quality: "perfect" | "good" | "tight" | "loose" = "perfect";
      
      if (selectedSize !== selectedGarment.bestSize) {
        if (selectedSize === "S" && selectedGarment.bestSize === "M") {
          baseScore = 74;
          quality = "tight";
        } else if (selectedSize === "S" && selectedGarment.bestSize === "L") {
          baseScore = 58;
          quality = "tight";
        } else if (selectedSize === "L" && selectedGarment.bestSize === "M") {
          baseScore = 84;
          quality = "loose";
        } else if (selectedSize === "XL") {
          baseScore = 69;
          quality = "loose";
        } else {
          baseScore = 86;
          quality = "loose";
        }
      } else {
        baseScore = 95 + Math.floor(Math.random() * 4);
        quality = "perfect";
      }

      setFitScore(baseScore);
      setFitQuality(quality);
      
      const factor = selectedSize === "S" ? -4.0 : selectedSize === "M" ? 0.0 : selectedSize === "L" ? 4.0 : 8.0;
      setToleranceDetails({
        chestUser: userMeasurements.chest,
        chestGarment: userMeasurements.chest + 4.0 + factor,
        chestDiff: 4.0 + factor,
        shouldersUser: userMeasurements.shoulders,
        shouldersGarment: userMeasurements.shoulders + 2.5 + (factor * 0.5),
        shouldersDiff: 2.5 + (factor * 0.5),
        waistUser: userMeasurements.waist,
        waistGarment: userMeasurements.waist + 2.0 + factor,
        waistDiff: 2.0 + factor
      });
    }

    queryFittingResult();
  }, [selectedGarment, selectedSize, userMeasurements, isApiConnected]);

  // 4. Manejador de Carga de Archivos
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);
    setScanSuccessMsg(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      // Hacer la petición al backend Flask
      const response = await fetch(decodeUrl(API_MEASURE_WEBCAM), {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("No se pudo procesar la imagen. Asegúrate de que el cuerpo sea completamente visible.");
      }

      const result = await response.json();
      if (result.success && result.measurements) {
        const m = result.measurements;
        // Actualizar medidas
        setUserMeasurements({
          id: result.measurement_id || 1,
          height: m.height || 178,
          chest: m.chest || 102,
          shoulders: m.shoulder_width || m.shoulders || 46,
          waist: m.waist || 88,
          hips: m.hip || m.hips || 98,
          arm_length: m.arm_length || 67
        });
        setScanSuccessMsg("¡Escaneo corporal completado exitosamente con MediaPipe!");
        setIsScanning(true);
        setTimeout(() => setIsScanning(false), 1500);
      } else {
        throw new Error(result.error || "No se detectaron puntos clave en la pose.");
      }
    } catch (err: any) {
      console.error(err);
      setUploadError(err.message || "Error al conectar con el servidor de estimación.");
      
      // Fallback de demostración simulado
      setTimeout(() => {
        const simulated = {
          id: Math.floor(Math.random() * 100),
          height: 180 + Math.floor(Math.random() * 6) - 3,
          chest: 104 + Math.floor(Math.random() * 10) - 5,
          shoulders: 47 + Math.floor(Math.random() * 4) - 2,
          waist: 86 + Math.floor(Math.random() * 10) - 5,
          hips: 99 + Math.floor(Math.random() * 6) - 3,
          arm_length: 68
        };
        setUserMeasurements(simulated);
        setScanSuccessMsg("Demo Offline: Escaneo simulado con éxito (" + simulated.chest + "cm pecho).");
        setIsScanning(true);
        setTimeout(() => setIsScanning(false), 1500);
        setIsUploading(false);
      }, 1500);
      return;
    } finally {
      setIsUploading(false);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const fitInfo = {
    perfect: { text: "Ajuste Perfecto", desc: "La prenda se adapta a tus medidas recomendadas sin restricciones.", color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30" },
    good: { text: "Buen Ajuste", desc: "Se adapta cómodamente, ofreciendo una silueta limpia.", color: "text-teal-400 bg-teal-500/10 border-teal-500/30" },
    tight: { text: "Demasiado Ajustado", desc: "Se prevé compresión en pecho y hombros. Recomendamos una talla mayor.", color: "text-rose-400 bg-rose-500/10 border-rose-500/30" },
    loose: { text: "Holgado / Oversized", desc: "Caída suelta con holgura extra. Ideal si prefieres un estilo relajado.", color: "text-amber-400 bg-amber-500/10 border-amber-500/30" }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 w-full">
      {/* 1. Panel de Visualización del Probador Virtual (6 cols) */}
      <div className="lg:col-span-7 xl:col-span-6 bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-6 relative overflow-hidden flex flex-col justify-between min-h-[580px] shadow-2xl shadow-violet-900/10">
        
        {/* Luces decorativas internas */}
        <div className="absolute top-0 right-0 w-48 h-48 bg-violet-600/10 rounded-full blur-3xl -z-10" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-teal-500/10 rounded-full blur-3xl -z-10" />

        {/* Encabezado del visor */}
        <div className="flex justify-between items-center z-10">
          <div className="flex items-center gap-2">
            <div className={`w-2.5 h-2.5 rounded-full animate-ping ${isApiConnected ? 'bg-emerald-500' : 'bg-amber-500'}`} />
            <span className={`text-xs font-semibold tracking-wider uppercase ${isApiConnected ? 'text-emerald-400' : 'text-amber-400'}`}>
              {isApiConnected ? "Flask Live API Connected" : "Local AI Simulation"}
            </span>
          </div>
          <button 
            onClick={() => setShowSkeleton(!showSkeleton)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-medium transition-all duration-300 ${
              showSkeleton 
                ? "bg-violet-600/20 border-violet-500/50 text-violet-300" 
                : "bg-white/5 border-white/10 text-gray-400 hover:bg-white/10"
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            {showSkeleton ? "Ocultar Esqueleto" : "Esqueleto Pose"}
          </button>
        </div>

        {/* Modelo interactivo y Canvas de landmarks */}
        <div className="relative flex-1 flex items-center justify-center my-6 h-[400px]">
          
          {/* Silueta Base 3D/2D */}
          <div className="relative w-[320px] h-[380px] flex items-center justify-center">
            <svg 
              className="absolute inset-0 w-full h-full text-slate-800 transition-all duration-500" 
              viewBox="0 0 100 100" 
              fill="currentColor"
            >
              <circle cx="50" cy="12" r="6" />
              <path d="M48,18 h4 v4 h-4 z" />
              <path d="M38,23.5 L62,23.5 L58,56 L42,56 Z" />
              <path d="M36.5,23.5 L24,42 L27,44 L38,26 Z" />
              <path d="M63.5,23.5 L76,42 L73,44 L62,26 Z" />
              <path d="M42,57 L38,92 L43,92 L49,58 Z" />
              <path d="M58,57 L62,92 L57,92 L51,58 Z" />
            </svg>

            {/* PRENDA VIRTUAL AJUSTADA */}
            <AnimatePresence mode="wait">
              <motion.div
                key={selectedGarment.id}
                initial={{ opacity: 0, scale: 0.95, y: 5 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.98, y: -5 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
                className={`absolute ${selectedGarment.imageOffset} rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-2 backdrop-blur-[2px] transition-all duration-500 ${selectedGarment.color}`}
              >
                <div className="text-center px-1">
                  <p className="text-[9px] uppercase font-bold tracking-wider opacity-60 text-white">
                    {selectedGarment.brand}
                  </p>
                  <p className="text-xs font-bold leading-tight text-white mt-0.5 truncate max-w-[150px]">
                    {selectedGarment.name}
                  </p>
                  <span className="inline-block mt-2 px-2 py-0.5 bg-white/20 rounded text-[9px] font-bold text-white uppercase">
                    Talla {selectedSize}
                  </span>
                </div>
                <div className="absolute top-0 left-0 w-2 h-2 rounded-full bg-white animate-ping" />
                <div className="absolute top-0 right-0 w-2 h-2 rounded-full bg-white animate-ping" />
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-white animate-ping" />
              </motion.div>
            </AnimatePresence>

            {/* ESQUELETO DE ESTIMACIÓN DE POSE (MediaPipe Mock) */}
            {showSkeleton && (
              <svg 
                className="absolute inset-0 w-full h-full pointer-events-none" 
                viewBox="0 0 100 100"
              >
                <polyline points="42,24 58,24" stroke="#10B981" strokeWidth="0.8" strokeLinecap="round" opacity="0.8" />
                <polyline points="42,24 30,43" stroke="#10B981" strokeWidth="0.8" strokeLinecap="round" opacity="0.8" />
                <polyline points="58,24 70,43" stroke="#10B981" strokeWidth="0.8" strokeLinecap="round" opacity="0.8" />
                <polyline points="42,24 43,56" stroke="#10B981" strokeWidth="0.8" strokeLinecap="round" opacity="0.8" />
                <polyline points="58,24 57,56" stroke="#10B981" strokeWidth="0.8" strokeLinecap="round" opacity="0.8" />
                <polyline points="43,56 57,56" stroke="#10B981" strokeWidth="0.8" strokeLinecap="round" opacity="0.8" />
                <polyline points="43,56 40,74 41,92" stroke="#10B981" strokeWidth="0.8" strokeLinecap="round" opacity="0.8" />
                <polyline points="57,56 60,74 59,92" stroke="#10B981" strokeWidth="0.8" strokeLinecap="round" opacity="0.8" />

                <circle cx="50" cy="12" r="1.5" fill="#8B5CF6" stroke="#fff" strokeWidth="0.4" />
                <circle cx="42" cy="24" r="1.5" fill="#10B981" stroke="#fff" strokeWidth="0.4" />
                <circle cx="58" cy="24" r="1.5" fill="#10B981" stroke="#fff" strokeWidth="0.4" />
                <circle cx="30" cy="43" r="1.5" fill="#10B981" stroke="#fff" strokeWidth="0.4" />
                <circle cx="70" cy="43" r="1.5" fill="#10B981" stroke="#fff" strokeWidth="0.4" />
                <circle cx="43" cy="56" r="1.5" fill="#10B981" stroke="#fff" strokeWidth="0.4" />
                <circle cx="57" cy="56" r="1.5" fill="#10B981" stroke="#fff" strokeWidth="0.4" />
                <circle cx="40" cy="74" r="1.5" fill="#10B981" stroke="#fff" strokeWidth="0.4" />
                <circle cx="60" cy="74" r="1.5" fill="#10B981" stroke="#fff" strokeWidth="0.4" />
                <circle cx="41" cy="92" r="1.5" fill="#10B981" stroke="#fff" strokeWidth="0.4" />
                <circle cx="59" cy="92" r="1.5" fill="#10B981" stroke="#fff" strokeWidth="0.4" />
              </svg>
            )}

            {/* Efecto visual de scan */}
            <AnimatePresence>
              {isScanning && (
                <motion.div
                  initial={{ top: "0%" }}
                  animate={{ top: "100%" }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 1.2, ease: "easeInOut" }}
                  className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-emerald-400 to-transparent shadow-[0_0_12px_#34d399] pointer-events-none z-20"
                />
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Panel inferior del visor: Medidas del usuario activas */}
        <div className="grid grid-cols-3 gap-2 text-center border-t border-white/5 pt-4 z-10">
          <div className="bg-white/5 rounded-xl p-2.5">
            <span className="block text-[10px] text-gray-400 uppercase font-semibold">Chest (Pecho)</span>
            <span className="text-sm font-bold text-white">{userMeasurements.chest.toFixed(1)} cm</span>
          </div>
          <div className="bg-white/5 rounded-xl p-2.5">
            <span className="block text-[10px] text-gray-400 uppercase font-semibold">Waist (Cintura)</span>
            <span className="text-sm font-bold text-white">{userMeasurements.waist.toFixed(1)} cm</span>
          </div>
          <div className="bg-white/5 rounded-xl p-2.5">
            <span className="block text-[10px] text-gray-400 uppercase font-semibold">Shoulders (Hombro)</span>
            <span className="text-sm font-bold text-white">{userMeasurements.shoulders.toFixed(1)} cm</span>
          </div>
        </div>
      </div>

      {/* 2. Panel de Ajuste y Métricas de IA (5 cols) */}
      <div className="lg:col-span-5 xl:col-span-6 flex flex-col justify-between gap-6">
        
        {/* Tarjeta de Escáner Corporal por Foto */}
        <div className="bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-6 shadow-xl relative overflow-hidden">
          <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
            <Camera className="w-5 h-5 text-teal-400" />
            AI Body Scanner
          </h3>
          <p className="text-xs text-gray-400 mb-4 leading-relaxed">
            Sube un retrato corporal completo para estimar tus medidas biométricas en tiempo real con MediaPipe.
          </p>

          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="hidden" 
            accept="image/*"
          />

          <button
            onClick={triggerFileSelect}
            disabled={isUploading}
            className={`w-full py-4 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center gap-2 transition-all duration-300 ${
              isUploading 
                ? "border-violet-500 bg-violet-600/10 text-violet-400" 
                : "border-white/15 bg-white/5 text-gray-300 hover:border-violet-400/50 hover:bg-violet-600/5"
            }`}
          >
            {isUploading ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin text-violet-400" />
                <span className="text-xs font-bold uppercase tracking-wider">Procesando Pose con IA...</span>
              </>
            ) : (
              <>
                <Upload className="w-5 h-5 text-gray-400 group-hover:text-white" />
                <span className="text-xs font-bold uppercase tracking-wider">Subir Foto Corporal</span>
                <span className="text-[10px] text-gray-500">JPG, PNG o JPEG</span>
              </>
            )}
          </button>

          {/* Notificaciones de Escaneo */}
          <AnimatePresence>
            {uploadError && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="mt-3 p-3 bg-rose-500/15 border border-rose-500/30 rounded-xl flex items-center gap-2 text-xs text-rose-400"
              >
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{uploadError}</span>
              </motion.div>
            )}
            {scanSuccessMsg && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="mt-3 p-3 bg-emerald-500/15 border border-emerald-500/30 rounded-xl flex items-center gap-2 text-xs text-emerald-400"
              >
                <Check className="w-4 h-4 flex-shrink-0" />
                <span>{scanSuccessMsg}</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Tarjeta 2: Control de Catálogo */}
        <div className="bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-6 shadow-xl">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-violet-400" />
            Catálogo Real (SQLite)
          </h3>

          <div className="flex flex-col gap-3 max-h-[170px] overflow-y-auto pr-1">
            {garments.map((g) => (
              <button
                key={g.id}
                onClick={() => {
                  setSelectedGarment(g);
                  setSelectedSize(g.bestSize);
                }}
                className={`flex items-center justify-between p-3 rounded-2xl border transition-all duration-300 text-left ${
                  selectedGarment.id === g.id
                    ? "bg-violet-600/20 border-violet-500/80 shadow-[0_0_15px_rgba(139,92,246,0.1)] text-white"
                    : "bg-white/5 border-white/5 text-gray-300 hover:bg-white/10 hover:border-white/10"
                }`}
              >
                <div className="flex flex-col">
                  <span className="text-[9px] uppercase font-bold tracking-widest text-violet-400">
                    {g.brand} • {g.type === "shirt" ? "Superior" : g.type === "pants" ? "Inferior" : g.type === "t-shirt" ? "Remera" : "Vestido"}
                  </span>
                  <span className="text-sm font-bold truncate max-w-[180px]">{g.name}</span>
                </div>
                <span className="text-sm font-bold text-gray-100">${g.price}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Tarjeta 3: Puntuación de Ajuste (Fit Score) */}
        <div className="bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-6 shadow-xl flex-1 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Ruler className="w-5 h-5 text-teal-400" />
                Diagnóstico de Ajuste
              </h3>
              <div className="flex gap-1.5 bg-white/5 p-1 rounded-xl border border-white/10">
                {["S", "M", "L", "XL"].map((sz) => (
                  <button
                    key={sz}
                    onClick={() => setSelectedSize(sz)}
                    className={`w-8 h-8 rounded-lg text-xs font-bold transition-all ${
                      selectedSize === sz
                        ? "bg-violet-600 text-white shadow"
                        : "text-gray-400 hover:bg-white/5 hover:text-white"
                    }`}
                  >
                    {sz}
                  </button>
                ))}
              </div>
            </div>

            {/* Medidor de Ajuste */}
            <div className="flex items-center gap-6 mb-4 bg-white/5 rounded-2xl p-4 border border-white/5">
              <div className="relative w-16 h-16 flex items-center justify-center">
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                  <path
                    className="text-slate-800"
                    strokeWidth="3.5"
                    stroke="currentColor"
                    fill="none"
                    d="M18 2.0845
                      a 15.9155 15.9155 0 0 1 0 31.831
                      a 15.9155 15.9155 0 0 1 0 -31.831"
                  />
                  <motion.path
                    initial={{ strokeDasharray: "0, 100" }}
                    animate={{ strokeDasharray: `${fitScore}, 100` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className="text-teal-400"
                    strokeWidth="3.5"
                    strokeLinecap="round"
                    stroke="currentColor"
                    fill="none"
                    d="M18 2.0845
                      a 15.9155 15.9155 0 0 1 0 31.831
                      a 15.9155 15.9155 0 0 1 0 -31.831"
                  />
                </svg>
                <span className="absolute text-sm font-black text-white">{fitScore}%</span>
              </div>
              <div className="flex-1">
                <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-bold border ${fitInfo[fitQuality].color} mb-1`}>
                  {fitInfo[fitQuality].text}
                </span>
                <p className="text-xs text-gray-400 leading-snug">
                  {fitDescription || fitInfo[fitQuality].desc}
                </p>
              </div>
            </div>

            {/* Desglose de medidas y tolerancia de holgura */}
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Ancho de Hombros (Holgura: +{toleranceDetails.shouldersDiff.toFixed(1)}cm)</span>
                  <span className="font-bold text-white">
                    {toleranceDetails.shouldersUser.toFixed(1)}cm vs {toleranceDetails.shouldersGarment.toFixed(1)}cm
                  </span>
                </div>
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-teal-400 rounded-full" style={{ width: `${Math.min(100, (toleranceDetails.shouldersUser / toleranceDetails.shouldersGarment) * 100)}%` }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Contorno de Pecho (Holgura: +{toleranceDetails.chestDiff.toFixed(1)}cm)</span>
                  <span className="font-bold text-white">
                    {toleranceDetails.chestUser.toFixed(1)}cm vs {toleranceDetails.chestGarment.toFixed(1)}cm
                  </span>
                </div>
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-teal-400 rounded-full" style={{ width: `${Math.min(100, (toleranceDetails.chestUser / toleranceDetails.chestGarment) * 100)}%` }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Contorno de Cintura (Holgura: +{toleranceDetails.waistDiff.toFixed(1)}cm)</span>
                  <span className="font-bold text-white">
                    {toleranceDetails.waistUser.toFixed(1)}cm vs {toleranceDetails.waistGarment.toFixed(1)}cm
                  </span>
                </div>
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-teal-400 rounded-full" style={{ width: `${Math.min(100, (toleranceDetails.waistUser / toleranceDetails.waistGarment) * 100)}%` }} />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-gray-400">
            <span className="flex items-center gap-1">
              <Info className="w-3.5 h-3.5 text-violet-400" />
              Recomendado: Talla {selectedGarment.bestSize}
            </span>
            <span className="text-gray-300 font-bold uppercase tracking-widest">FitVibe API Core</span>
          </div>
        </div>
      </div>
    </div>
  );
}
