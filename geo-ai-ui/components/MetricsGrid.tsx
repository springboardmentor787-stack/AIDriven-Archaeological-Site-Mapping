import { 
  Wind, 
  Mountain, 
  Droplets, 
  Navigation, 
  Layers, 
  Box, 
  History, 
  ArrowUpRight,
  Compass
} from "lucide-react";

type Metric = {
  label: string;
  value: string;
};

type MetricsGridProps = {
  metrics: Metric[];
};

const ICON_BY_METRIC: Record<string, any> = {
  Vegetation: Wind,
  Slope: Mountain,
  Rainfall: Droplets,
  Elevation: Navigation,
  Soil: Layers,
  Boulders: Box,
  Ruins: History,
  Structures: ArrowUpRight,
  Coordinates: Compass,
};

export default function MetricsGrid({ metrics }: MetricsGridProps) {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
      {metrics.map((metric) => {
        const Icon = ICON_BY_METRIC[metric.label] || Compass;
        return (
          <div 
            key={metric.label} 
            className="group glass-card bg-white/[0.01] p-5 transition-all duration-300 hover:bg-white/[0.04] hover:border-brand-500/30 hover:scale-[1.02]"
          >
            <div className="mb-3 flex items-center justify-between">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-500/10 border border-brand-500/10 group-hover:border-brand-500/30 transition-colors">
                <Icon className="h-4 w-4 text-brand-400" />
              </div>
              <div className="h-1 w-1 rounded-full bg-brand-500/20 group-hover:bg-brand-500 transition-colors" />
            </div>
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-dim/60 transition-colors group-hover:text-brand-300/80">
              {metric.label}
            </p>
            <p className="mt-1 font-[var(--font-sora)] text-sm font-bold text-ink group-hover:text-brand-50 transition-colors">
              {metric.value}
            </p>
          </div>
        );
      })}
    </div>
  );
}
