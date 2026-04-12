"use client";

import Image from "next/image";
import { 
  UploadCloud, 
  Settings2, 
  Map as MapIcon, 
  Zap, 
  FileText, 
  History as HistoryIcon,
  Globe,
  Loader2,
  Trash2
} from "lucide-react";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import ExpandableSection from "@/components/ExpandableSection";
import ImageGrid from "@/components/ImageGrid";
import InsightBox from "@/components/InsightBox";
import MetricsGrid from "@/components/MetricsGrid";
import ResultCard from "@/components/ResultCard";

const MapPicker = dynamic(() => import("@/components/MapPicker"), { ssr: false });

type RiskLabel = "LOW" | "MODERATE" | "HIGH";

type PredictionResult = {
  probability: number;
  explanation: string;
  insightMode?: "groq" | "local" | "default";
  insightStatus?: string;
  metrics: {
    vegetation: number;
    slope: number;
    rainfall: number;
    elevation: number;
    soil: string;
    boulders: number;
    ruins: number;
    structures: number;
    lat: number;
    lon: number;
  };
  shap: { feature: string; value: number }[];
  images: {
    original: string;
    detection: string;
    segmentation: string;
    combined: string;
    heatmap: string;
  };
};

type HistoryItem = {
  id: string;
  timestamp: string;
  probability: number;
  lat: number;
  lon: number;
  risk: RiskLabel;
};

const STAGES = [
  "Initializing scan...",
  "Acquiring satellite data...",
  "Running neural detection...",
  "Segmenting terrain features...",
  "Calculating erosion vectors...",
  "Finalizing AI insights..."
];

function getRiskLabel(probability: number): RiskLabel {
  if (probability < 0.3) return "LOW";
  if (probability < 0.7) return "MODERATE";
  return "HIGH";
}

function shapInsight(feature: string, value: number): string {
  const key = feature.toLowerCase();
  const intensity = Math.abs(value) > 1 ? "strongly" : Math.abs(value) > 0.45 ? "moderately" : "slightly";

  if (key.includes("vegetation")) {
    return value < 0
      ? `Vegetation index ${intensity} mitigates erosion potential.`
      : `Sparse vegetation ${intensity} elevates surface vulnerability.`;
  }
  if (key.includes("slope")) {
    return value >= 0
      ? `Steep gradient ${intensity} accelerates runoff risks.`
      : `Favorable slope ${intensity} promotes terrain stability.`;
  }
  if (key.includes("rain")) {
    return value >= 0
      ? `Precipitation levels ${intensity} increase hydraulic pressure.`
      : `Hydrological context ${intensity} reduces erosion force.`;
  }
  if (key.includes("elevation")) {
    return value >= 0
      ? `Altitude context ${intensity} drives risk elevation.`
      : `Elevation baseline ${intensity} stabilizes the site.`;
  }
  return value >= 0
    ? `${feature} analysis ${intensity} increases hazard score.`
    : `${feature} analysis ${intensity} reduces hazard score.`;
}

export default function Page() {
  const [lat, setLat] = useState("22.5726");
  const [lon, setLon] = useState("88.3639");
  const [apiKey, setApiKey] = useState("");
  const [useCustomApiKey, setUseCustomApiKey] = useState(false);
  const [uploadUrl, setUploadUrl] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [confidence, setConfidence] = useState(0.5);
  const [showVegetation, setShowVegetation] = useState(true);
  const [showRuins, setShowRuins] = useState(true);
  const [showStructures, setShowStructures] = useState(true);
  const [showBoulders, setShowBoulders] = useState(true);
  const [showOthers, setShowOthers] = useState(true);
  const [useAiInsight, setUseAiInsight] = useState(true);
  const [loading, setLoading] = useState(false);
  const [stageIndex, setStageIndex] = useState(0);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem("geo-ai-history");
      if (raw) {
        const parsed = JSON.parse(raw) as HistoryItem[];
        if (Array.isArray(parsed)) setHistory(parsed.slice(0, 8));
      }
    } catch {}
  }, []);

  useEffect(() => {
    if (!loading) {
      setStageIndex(0);
      return;
    }
    const interval = setInterval(() => {
      setStageIndex((prev) => (prev + 1) % STAGES.length);
    }, 1200);
    return () => clearInterval(interval);
  }, [loading]);

  const imageItems = useMemo(() => {
    if (!result?.images) return uploadUrl ? [{ label: "Source", src: uploadUrl }] : [];
    return [
      { label: "Original", src: result.images.original },
      { label: "Detection", src: result.images.detection },
      { label: "Segmentation", src: result.images.segmentation },
      { label: "Combined", src: result.images.combined },
      { label: "Heatmap", src: result.images.heatmap }
    ];
  }, [result, uploadUrl]);

  const metrics = useMemo(() => {
    if (!result) return [];
    return [
      { label: "Vegetation", value: `${(result.metrics.vegetation * 100).toFixed(1)}%` },
      { label: "Slope", value: `${result.metrics.slope.toFixed(1)}°` },
      { label: "Rainfall", value: `${result.metrics.rainfall.toFixed(0)} mm` },
      { label: "Elevation", value: `${result.metrics.elevation.toFixed(0)} m` },
      { label: "Soil Type", value: result.metrics.soil },
      { label: "Boulders", value: result.metrics.boulders.toFixed(3) },
      { label: "Archaeological", value: result.metrics.ruins.toFixed(3) },
      { label: "Structures", value: result.metrics.structures.toFixed(3) },
      { label: "Coordinates", value: `${result.metrics.lat.toFixed(4)}, ${result.metrics.lon.toFixed(4)}` }
    ];
  }, [result]);

  async function runPrediction() {
    if (!selectedFile) return setError("Satellite source required for analysis.");
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      formData.append("lat", lat);
      formData.append("lon", lon);
      if (useCustomApiKey && apiKey.trim()) formData.append("apiKey", apiKey.trim());
      formData.append("confidence", confidence.toString());
      formData.append("showVegetation", String(showVegetation));
      formData.append("showRuins", String(showRuins));
      formData.append("showStructures", String(showStructures));
      formData.append("showBoulders", String(showBoulders));
      formData.append("useAiInsight", String(useAiInsight));

      const res = await fetch("/api/predict", { method: "POST", body: formData });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "System failure during prediction.");
      
      const resData = payload as PredictionResult;
      setResult(resData);
      
      const item: HistoryItem = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        probability: resData.probability,
        lat: Number(lat),
        lon: Number(lon),
        risk: getRiskLabel(resData.probability)
      };
      setHistory((prev) => {
        const next = [item, ...prev].slice(0, 8);
        window.localStorage.setItem("geo-ai-history", JSON.stringify(next));
        return next;
      });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function downloadPdfReport() {
    if (!result) return;
    const { jsPDF } = await import("jspdf");
    const doc = new jsPDF();
    doc.setFontSize(20);
    doc.text("Geo AI System: Terrain Analysis Report", 20, 20);
    doc.setFontSize(12);
    doc.text(`Timestamp: ${new Date().toLocaleString()}`, 20, 30);
    doc.text(`Location: ${lat}, ${lon}`, 20, 40);
    doc.text(`Hazard Probability: ${(result.probability * 100).toFixed(1)}%`, 20, 50);
    doc.text("AI Explanation:", 20, 65);
    const splitText = doc.splitTextToSize(result.explanation, 170);
    doc.text(splitText, 20, 75);
    doc.save(`GeoAI_Report_${Date.now()}.pdf`);
  }

  const onUploadFile = async (file?: File) => {
    if (!file) return;
    setSelectedFile(file);
    setUploadUrl(URL.createObjectURL(file));
    setError("");
    try {
      const exifr = await import("exifr");
      const gps = await exifr.gps(file);
      if (gps?.latitude && gps?.longitude) {
        setLat(gps.latitude.toFixed(6));
        setLon(gps.longitude.toFixed(6));
      }
    } catch {}
  };

  return (
    <main className="mx-auto min-h-screen w-full max-w-[1440px] px-4 py-8 md:px-12 md:py-16">
      {/* Header Section */}
      <header className="mb-12 flex flex-col items-start justify-between gap-6 md:flex-row md:items-end animate-fadeUp">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="h-2 w-12 rounded-full bg-brand-500 shadow-glow" />
            <span className="text-[10px] font-black uppercase tracking-[0.4em] text-brand-500/80">Mission Intelligence</span>
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-ink md:text-5xl lg:text-6xl">
            Geo <span className="text-brand-500 text-glow">AI</span> System
          </h1>
          <p className="max-w-xl text-sm font-medium text-dim/80 leading-relaxed md:text-base">
            Advanced neural mapping for archaeological site preservation and environmental erosion monitoring.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex flex-col items-end mr-4">
             <span className="text-[9px] font-bold uppercase tracking-widest text-dim/40">Status</span>
             <span className="flex items-center gap-2 text-xs font-bold text-ink">
                <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]" />
                SYSTEM ACTIVE
             </span>
          </div>
          <button className="button-secondary gap-2" onClick={() => window.location.reload()}>
            <RefreshCw size={16} />
            Reset
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[400px_1fr]">
        {/* Left Column: Command & Controls */}
        <aside className="space-y-6">
          <div className="glass-panel p-6 sticky top-8">
            <div className="mb-6 flex items-center gap-3 border-b border-white/5 pb-4">
                <Settings2 className="h-5 w-5 text-brand-400" />
                <h2 className="font-[var(--font-sora)] text-sm font-black uppercase tracking-widest text-ink">Scan Parameters</h2>
            </div>

            <div className="space-y-6">
              {/* Image Upload */}
              <div className="space-y-3">
                <label className="text-[10px] font-bold uppercase tracking-widest text-dim/60">Satellite Imagery Source</label>
                <div className="group relative overflow-hidden rounded-2xl border-2 border-dashed border-white/5 bg-white/[0.02] transition-all hover:border-brand-500/20 hover:bg-white/[0.04]">
                  {uploadUrl ? (
                    <div className="relative h-48 w-full">
                      <Image src={uploadUrl} alt="Preview" fill className="object-cover transition-transform duration-700 group-hover:scale-105" />
                      <div className="absolute inset-0 bg-black/40 opacity-0 transition-opacity group-hover:opacity-100 flex items-center justify-center">
                         <button onClick={() => {setUploadUrl(""); setSelectedFile(null)}} className="h-10 w-10 rounded-full bg-high/20 border border-high/30 flex items-center justify-center text-high hover:bg-high/40 transition-colors">
                            <Trash2 size={20} />
                         </button>
                      </div>
                    </div>
                  ) : (
                    <label className="flex h-40 cursor-pointer flex-col items-center justify-center gap-3">
                      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand-500/5 border border-brand-500/10 text-brand-400">
                        <UploadCloud size={24} />
                      </div>
                      <span className="text-xs font-bold text-dim/80">Drop Satellite Image</span>
                      <input type="file" accept="image/*" className="hidden" onChange={(e) => onUploadFile(e.target.files?.[0])} />
                    </label>
                  )}
                </div>
              </div>

              {/* Coordinates */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-[10px] font-bold uppercase tracking-widest text-dim/60 italic">LATITUDE</label>
                  <input className="input font-mono" value={lat} onChange={(e) => setLat(e.target.value)} placeholder="0.0000" />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-bold uppercase tracking-widest text-dim/60 italic">LONGITUDE</label>
                  <input className="input font-mono" value={lon} onChange={(e) => setLon(e.target.value)} placeholder="0.0000" />
                </div>
              </div>

              {/* Map Interaction */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <label className="text-[10px] font-bold uppercase tracking-widest text-dim/60">Geospatial Selector</label>
                    <Globe className="h-3 w-3 text-brand-500/50" />
                </div>
                <div className="overflow-hidden rounded-2xl border border-white/5 shadow-2xl">
                    <MapPicker
                        lat={Number(lat) || 0}
                        lon={Number(lon) || 0}
                        onChange={(nLat, nLon) => {
                            setLat(nLat.toFixed(6));
                            setLon(nLon.toFixed(6));
                        }}
                    />
                </div>
              </div>

              {/* Model Confidence Slider */}
              <div className="space-y-4 rounded-2xl bg-white/[0.03] p-4 border border-white/5">
                <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-dim/60">
                    <span>Threshold Confidence</span>
                    <span className="text-brand-400">{(confidence * 100).toFixed(0)}%</span>
                </div>
                <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={confidence}
                    onChange={(e) => setConfidence(Number(e.target.value))}
                    className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-brand-500"
                />
                
                <div className="grid grid-cols-2 gap-y-3 pt-2">
                    {[
                        { id: "veg", label: "Vegetation", state: showVegetation, set: setShowVegetation },
                        { id: "ruin", label: "Ruins", state: showRuins, set: setShowRuins },
                        { id: "struct", label: "Structures", state: showStructures, set: setShowStructures },
                        { id: "bould", label: "Boulders", state: showBoulders, set: setShowBoulders },
                    ].map((opt) => (
                        <label key={opt.id} className="flex cursor-pointer items-center gap-2 group">
                            <div className="relative flex items-center">
                                <input 
                                    type="checkbox" 
                                    checked={opt.state} 
                                    onChange={(e) => opt.set(e.target.checked)} 
                                    className="peer h-4 w-4 cursor-pointer appearance-none rounded border border-white/10 bg-white/5 transition-all checked:bg-brand-500 checked:border-brand-500" 
                                />
                                <div className="pointer-events-none absolute left-1 opacity-0 peer-checked:opacity-100 text-surface transition-opacity">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4" className="h-2 w-2"><path d="M5 13l4 4L19 7" /></svg>
                                </div>
                            </div>
                            <span className="text-[11px] font-bold text-dim transition-colors group-hover:text-ink">{opt.label}</span>
                        </label>
                    ))}
                </div>
              </div>

              {/* Action Button */}
              <button 
                className="button-primary group w-full gap-3 py-4" 
                onClick={runPrediction} 
                disabled={loading}
              >
                {loading ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                    <Zap className="h-5 w-5 fill-brand-500 group-hover:scale-110 transition-transform" />
                )}
                <span className="tracking-[0.1em]">{loading ? "Processing..." : "START SYSTEM SCAN"}</span>
              </button>

              {error && (
                <div className="flex gap-3 rounded-xl bg-high/10 border border-high/20 p-3 text-[11px] font-bold text-high uppercase tracking-wider animate-fadeUp">
                   <AlertTriangle size={14} className="shrink-0" />
                   {error}
                </div>
              )}
            </div>
          </div>
        </aside>

        {/* Right Column: Results & Analytics */}
        <section className="space-y-8 min-w-0">
          {/* Default State */}
          {!result && !loading && (
             <div className="glass-panel flex min-h-[500px] flex-col items-center justify-center p-12 text-center animate-fadeUp">
                <div className="relative mb-8">
                   <div className="absolute inset-0 animate-ping rounded-full bg-brand-500/20" />
                   <div className="relative flex h-24 w-24 items-center justify-center rounded-full bg-brand-500/10 border border-brand-500/20">
                      <MapIcon className="h-10 w-10 text-brand-500" />
                   </div>
                </div>
                <h3 className="font-[var(--font-sora)] text-2xl font-bold text-ink">System Identification Required</h3>
                <p className="mt-4 max-w-md text-dim/70 leading-relaxed font-medium">
                   Interface is ready for terrain data input. Please upload a satellite capture and define location coordinates to begin the AI-driven hazard assessment.
                </p>
                <div className="mt-8 flex gap-4">
                   <div className="flex flex-col items-center gap-1">
                      <div className="h-1 w-12 rounded-full bg-white/5" />
                      <span className="text-[9px] font-bold text-dim/40 tracking-widest uppercase">Buffer Clear</span>
                   </div>
                </div>
             </div>
          )}

          {/* Loading State */}
          {loading && (
             <div className="glass-panel flex min-h-[300px] flex-col items-center justify-center p-12 text-center animate-fadeUp">
                <div className="mb-8 flex flex-col items-center gap-4">
                   <div className="h-1 w-1 animate-ping rounded-full bg-brand-500" />
                   <h3 className="bg-gradient-to-r from-ink to-dim bg-clip-text font-[var(--font-sora)] text-4xl font-black text-transparent md:text-5xl lg:text-6xl">
                      {STAGES[stageIndex]}
                   </h3>
                </div>
                <div className="h-2 w-full max-w-2xl overflow-hidden rounded-full bg-white/[0.03] ring-1 ring-white/5">
                   <div className="h-full w-full origin-left animate-shimmer bg-brand-500/40 shadow-glow" />
                </div>
                <p className="mt-8 text-[10px] font-black uppercase tracking-[0.4em] text-brand-500/60">Primary Processing active</p>
             </div>
          )}

          {/* Results Grid */}
          {result && (
            <div className="space-y-8 animate-fadeUp">
              <div className="grid grid-cols-1 gap-8 xl:grid-cols-[1fr_400px]">
                  <ResultCard probability={result.probability} />
                  <div className="space-y-6">
                      <button className="button-primary w-full gap-2 bg-brand-600 text-white" onClick={downloadPdfReport}>
                         <FileText size={18} />
                         GENERATE PDF INTEL
                      </button>
                      <InsightBox 
                        text={result.explanation} 
                        loading={loading} 
                        risk={getRiskLabel(result.probability)}
                        mode={result.insightMode}
                        status={result.insightStatus}
                      />
                  </div>
              </div>

              <div className="grid grid-cols-1 gap-6">
                <ExpandableSection title="GEOSPATIAL METRIC DATA" subtitle="Raw input feature breakdown">
                   <MetricsGrid metrics={metrics} />
                </ExpandableSection>

                <ExpandableSection title="VISUAL SPECTROMETRY" subtitle="Neural layer visualization & feature detection">
                   <ImageGrid items={imageItems} />
                </ExpandableSection>

                <ExpandableSection title="HAZARD EXPLAINABILITY (SHAP)" subtitle="Feature importance & system rationale">
                  <div className="grid gap-6">
                    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                        {result.shap.map((item) => (
                          <div 
                            key={item.feature} 
                            className={`glass-card group p-5 border-l-4 transition-all hover:scale-[1.02] ${
                                item.value >= 0 ? "border-l-high bg-high/5" : "border-l-low bg-low/5"
                            }`}
                          >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-[10px] font-bold uppercase tracking-widest text-dim/60 group-hover:text-ink transition-colors">
                                    {item.feature}
                                </span>
                                <span className={`text-[10px] font-bold ${item.value >= 0 ? 'text-high' : 'text-low'}`}>
                                    {item.value >= 0 ? '+' : ''}{(item.value * 100).toFixed(1)}%
                                </span>
                            </div>
                            <p className="text-sm font-medium leading-relaxed text-ink/80">
                              {shapInsight(item.feature, item.value)}
                            </p>
                          </div>
                        ))}
                    </div>
                    
                    {/* Visual Feature Bar Chart */}
                    <div className="glass-card bg-white/[0.01] p-6">
                       <h4 className="mb-6 text-xs font-bold uppercase tracking-[0.2em] text-dim/70">Probability Contribution Matrix</h4>
                       <div className="space-y-6">
                          {result.shap.map((item) => {
                             const intensity = Math.min(100, Math.abs(item.value) * 160);
                             return (
                                <div key={`bar-${item.feature}`} className="space-y-2">
                                   <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-widest">
                                      <span className="text-ink">{item.feature}</span>
                                      <span className={item.value >= 0 ? "text-high" : "text-low"}>
                                         {item.value >= 0 ? "Critical Influence" : "Stabilizing Factor"}
                                      </span>
                                   </div>
                                   <div className="h-1.5 w-full rounded-full bg-white/[0.03] overflow-hidden">
                                      <div 
                                         className={`h-full rounded-full transition-all duration-1000 shadow-glow ${item.value >= 0 ? "bg-high" : "bg-low"}`}
                                         style={{ width: `${intensity}%` }}
                                      />
                                   </div>
                                </div>
                             )
                          })}
                       </div>
                    </div>
                  </div>
                </ExpandableSection>

                {history.length > 0 && (
                   <ExpandableSection title="RETROSPECTIVE LOGS" subtitle="Previous session activity (last 8 runs)">
                      <div className="grid gap-3">
                         {history.map((item) => (
                            <div key={item.id} className="glass-card flex items-center justify-between bg-white/[0.02] px-6 py-4 transition-all hover:bg-white/[0.05] hover:border-white/10 group">
                               <div className="flex items-center gap-6">
                                  <div className={`h-2 w-2 rounded-full ${item.risk === 'HIGH' ? 'bg-high shadow-[0_0_8px_rgba(239,68,68,0.8)]' : item.risk === 'MODERATE' ? 'bg-moderate' : 'bg-low'}`} />
                                  <div className="flex flex-col">
                                     <span className="text-[10px] font-bold text-ink uppercase tracking-widest">Run ID {item.id.slice(-6)}</span>
                                     <span className="text-[10px] font-medium text-dim/60 italic">{new Date(item.timestamp).toLocaleString()}</span>
                                  </div>
                                  <div className="hidden lg:flex flex-col">
                                     <span className="text-[9px] font-bold text-dim/40 uppercase tracking-widest">Coordinates</span>
                                     <span className="text-[10px] font-mono text-dim">{item.lat.toFixed(4)}, {item.lon.toFixed(4)}</span>
                                  </div>
                               </div>
                               <div className="flex items-center gap-6">
                                  <div className="flex flex-col items-end">
                                     <span className="text-[10px] font-black text-ink">{(item.probability * 100).toFixed(1)}%</span>
                                     <span className={`text-[9px] font-black uppercase tracking-widest ${item.probability > 0.7 ? 'text-high' : item.probability > 0.3 ? 'text-moderate' : 'text-low'}`}>
                                        {item.risk}
                                     </span>
                                  </div>
                                  <ArrowRight size={14} className="text-dim group-hover:translate-x-1 transition-transform" />
                               </div>
                            </div>
                         ))}
                      </div>
                   </ExpandableSection>
                )}
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

import { RefreshCw, ArrowRight, AlertTriangle } from "lucide-react";
