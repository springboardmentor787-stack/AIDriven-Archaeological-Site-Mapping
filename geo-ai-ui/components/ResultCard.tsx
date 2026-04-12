import { AlertTriangle, CheckCircle, Info } from "lucide-react";

type ResultCardProps = {
  probability: number;
};

function getRisk(probability: number) {
  if (probability < 0.3) return { 
    label: "LOW HAZARD", 
    color: "text-low", 
    bg: "bg-low/10", 
    border: "border-low/20",
    icon: CheckCircle,
    desc: "The terrain appears stable with minimal erosion markers."
  };
  if (probability < 0.7) return { 
    label: "MODERATE RISK", 
    color: "text-moderate", 
    bg: "bg-moderate/10", 
    border: "border-moderate/20",
    icon: Info,
    desc: "Active erosion detected. Surface stability is compromised."
  };
  return { 
    label: "HIGH CRITICAL", 
    color: "text-high", 
    bg: "bg-high/10", 
    border: "border-high/20",
    icon: AlertTriangle,
    desc: "Severe erosion threat. Immediate site stabilization required."
  };
}

export default function ResultCard({ probability }: ResultCardProps) {
  const risk = getRisk(probability);
  const percentage = Math.max(0, Math.min(100, Math.round(probability * 100)));
  const Icon = risk.icon;

  return (
    <section className="glass-panel relative overflow-hidden animate-fadeUp">
      {/* Decorative background glow */}
      <div className={`absolute -right-20 -top-20 h-64 w-64 rounded-full blur-[100px] opacity-20 ${risk.label.includes("LOW") ? "bg-low" : risk.label.includes("MODERATE") ? "bg-moderate" : "bg-high"}`} />
      
      <div className="relative flex flex-col items-center p-8 md:p-12">
        <header className="mb-6 flex flex-col items-center">
          <div className={`mb-4 flex h-12 w-12 items-center justify-center rounded-2xl ${risk.bg} ${risk.border} border`}>
            <Icon className={`h-6 w-6 ${risk.color}`} />
          </div>
          <span className={`risk-badge ${risk.bg} ${risk.color} border ${risk.border}`}>
            {risk.label}
          </span>
        </header>

        <div className="relative flex items-baseline justify-center">
          <span className={`font-[var(--font-sora)] text-8xl font-bold tracking-tighter md:text-9xl ${risk.color} text-glow`}>
            {percentage}
          </span>
          <span className={`ml-2 text-2xl font-bold md:text-3xl ${risk.color} opacity-60`}>%</span>
        </div>

        <p className="mt-4 max-w-sm text-center text-sm font-medium leading-relaxed text-dim">
          {risk.desc}
        </p>

        <footer className="mt-10 w-full max-w-lg">
          <div className="mb-3 flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-dim/60">
            <span>Probability Metric</span>
            <span>{percentage}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-white/[0.03] ring-1 ring-white/5">
            <div
              className={`h-full rounded-full transition-all duration-1000 ease-out shadow-glow ${
                risk.label.includes("LOW") ? "bg-low" : risk.label.includes("MODERATE") ? "bg-moderate" : "bg-high"
              }`}
              style={{ width: `${percentage}%` }}
            />
          </div>
        </footer>
      </div>
    </section>
  );
}
