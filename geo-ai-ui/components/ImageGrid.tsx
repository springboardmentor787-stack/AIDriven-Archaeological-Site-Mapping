"use client";

import Image from "next/image";
import { useState } from "react";
import { Maximize2, X } from "lucide-react";

type ImageItem = {
  label: string;
  src: string;
};

type ImageGridProps = {
  items: ImageItem[];
};

export default function ImageGrid({ items }: ImageGridProps) {
  const [active, setActive] = useState<ImageItem | null>(null);

  if (items.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center rounded-2xl border border-dashed border-white/10 bg-white/[0.02]">
        <p className="text-sm font-medium text-dim/50 italic">Waiting for visual analysis data...</p>
      </div>
    );
  }

  const orderedLabels = ["Original", "Detection", "Segmentation", "Combined"];
  const primaryItems = orderedLabels
    .map((label) => items.find((item) => item.label.toLowerCase() === label.toLowerCase()))
    .filter((item): item is ImageItem => Boolean(item));
  const heatmapItem = items.find((item) => item.label.toLowerCase() === "heatmap");

  return (
    <>
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {primaryItems.map((item) => (
          <figure 
            key={item.label} 
            className="group relative overflow-hidden rounded-2xl border border-white/10 bg-black/40 transition-all duration-500 hover:border-brand-500/30 hover:shadow-glow"
          >
            <button 
              type="button" 
              className="relative block aspect-[16/10] w-full overflow-hidden" 
              onClick={() => setActive(item)}
            >
              <Image 
                src={item.src} 
                alt={item.label} 
                fill 
                unoptimized 
                className="object-cover transition-transform duration-700 group-hover:scale-105" 
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
              <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between opacity-0 translate-y-2 transition-all duration-500 group-hover:opacity-100 group-hover:translate-y-0">
                <span className="text-[10px] font-bold uppercase tracking-widest text-white">{item.label} Analysis</span>
                <Maximize2 className="h-4 w-4 text-white" />
              </div>
            </button>
            <figcaption className="absolute left-3 top-3 rounded-lg bg-black/60 px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest text-brand-100 backdrop-blur-md border border-white/10">
              {item.label}
            </figcaption>
          </figure>
        ))}
      </div>

      {heatmapItem ? (
        <figure className="group relative mt-6 overflow-hidden rounded-2xl border border-white/10 bg-black/40 transition-all duration-500 hover:border-brand-500/30 hover:shadow-glow">
          <button 
            type="button" 
            className="relative block aspect-[21/8] w-full overflow-hidden" 
            onClick={() => setActive(heatmapItem)}
          >
            <Image 
              src={heatmapItem.src} 
              alt={heatmapItem.label} 
              fill 
              unoptimized 
              className="object-cover transition-transform duration-700 group-hover:scale-102" 
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
            <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between opacity-0 translate-y-2 transition-all duration-500 group-hover:opacity-100 group-hover:translate-y-0">
                <span className="text-[10px] font-bold uppercase tracking-widest text-white">Heatmap Intensity Map</span>
                <Maximize2 className="h-4 w-4 text-white" />
            </div>
          </button>
          <figcaption className="absolute left-3 top-3 rounded-lg bg-black/60 px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest text-brand-100 backdrop-blur-md border border-white/10">
            Probability Heatmap
          </figcaption>
        </figure>
      ) : null}

      {/* Modern Fullscreen Lightbox */}
      {active ? (
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center bg-surface/95 backdrop-blur-xl animate-fadeUp p-4" 
          onClick={() => setActive(null)}
        >
          <div className="relative h-[90vh] w-full max-w-7xl animate-float" onClick={(e) => e.stopPropagation()}>
            <div className="absolute -top-14 left-0 right-0 flex items-center justify-between text-white">
               <div className="flex flex-col">
                  <span className="text-[10px] font-bold uppercase tracking-[0.3em] text-brand-400">Analysis Preview</span>
                  <h3 className="font-[var(--font-sora)] text-xl font-bold">{active.label}</h3>
               </div>
               <button
                type="button"
                className="flex h-10 w-10 items-center justify-center rounded-full bg-white/10 border border-white/10 transition-colors hover:bg-white/20"
                onClick={() => setActive(null)}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="relative h-full w-full overflow-hidden rounded-3xl border border-white/10 shadow-2xl">
              <Image src={active.src} alt={active.label} fill unoptimized className="object-contain bg-black/20" />
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
