import { Brain, CornerDownRight, Sparkles } from "lucide-react";

type InsightBoxProps = {
  text: string;
  loading?: boolean;
  risk?: "LOW" | "MODERATE" | "HIGH";
  mode?: "groq" | "local" | "default";
  status?: string;
};

export default function InsightBox({ text, loading = false, risk = "MODERATE", mode = "default", status }: InsightBoxProps) {
  const riskColor =
    risk === "LOW"
      ? "text-low"
      : risk === "MODERATE"
      ? "text-moderate"
      : "text-high";

  return (
    <section className="glass-card bg-mesh-gradient group relative overflow-hidden p-6 md:p-8 animate-fadeUp">
      <div className="absolute inset-0 bg-shimmer opacity-20" />
      
      <header className="relative mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-500/10 border border-brand-500/20 shadow-glow">
            <Brain className="h-5 w-5 text-brand-400" />
          </div>
          <div>
            <h3 className="font-[var(--font-sora)] text-sm font-bold tracking-tight text-ink">AI Terrain Intelligence</h3>
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-dim/60">Generated Insight Report</p>
          </div>
        </div>

        {mode === "local" ? (
          <div className="flex items-center gap-1.5 rounded-full bg-amber-500/10 border border-amber-500/20 px-3 py-1 text-[10px] font-bold text-amber-400 uppercase tracking-widest">
            <Sparkles className="h-3 w-3" />
            Local Engine
          </div>
        ) : (
             <div className="flex items-center gap-1.5 rounded-full bg-brand-500/10 border border-brand-500/20 px-3 py-1 text-[10px] font-bold text-brand-400 uppercase tracking-widest text-glow">
                Live Cloud AI
             </div>
        )}
      </header>

      <div className="relative">
        {loading ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
                <div className="h-2 w-2 animate-pulse rounded-full bg-brand-500" />
                <p className="text-xs font-bold text-dim uppercase tracking-widest">Synthesizing landscape data...</p>
            </div>
            <div className="space-y-2">
                <div className="h-3 w-full animate-pulseSoft rounded bg-white/10" />
                <div className="h-3 w-11/12 animate-pulseSoft rounded bg-white/10" />
                <div className="h-3 w-4/5 animate-pulseSoft rounded bg-white/10" />
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex gap-3">
                <CornerDownRight className="mt-1 h-4 w-4 shrink-0 text-brand-500/40" />
                <p className="text-sm leading-relaxed text-ink/90 font-medium md:text-[15px]">
                    {text || "Analysis pending. Please initiate the prediction sequence to generate insights."}
                </p>
            </div>
            
            {mode === "local" && status ? (
              <div className="mt-4 flex items-start gap-2 rounded-xl bg-amber-950/20 border border-amber-500/10 p-3">
                <Info size={14} className="mt-0.5 shrink-0 text-amber-500" />
                <p className="text-[11px] font-medium leading-normal text-amber-200/70 italic">
                  Engine Note: {status}
                </p>
              </div>
            ) : null}

            <div className="pt-4 border-t border-white/5 flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="flex flex-col">
                        <span className="text-[9px] font-bold uppercase tracking-widest text-dim/50">Primary Indicator</span>
                        <span className={`text-xs font-bold ${riskColor}`}>DEGRADATION {risk}</span>
                    </div>
                </div>
                <div className="h-1.5 w-24 rounded-full bg-white/5 overflow-hidden">
                    <div className={`h-full ${riskColor === 'text-low' ? 'bg-low' : riskColor === 'text-moderate' ? 'bg-moderate' : 'bg-high'} opacity-40 shadow-glow`} style={{ width: '60%' }} />
                </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

import { Info } from "lucide-react";
