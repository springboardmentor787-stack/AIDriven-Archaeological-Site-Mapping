import { ChevronDown } from "lucide-react";
import { ReactNode, useState } from "react";

type ExpandableSectionProps = {
  title: string;
  subtitle?: string;
  children: ReactNode;
};

export default function ExpandableSection({ title, subtitle, children }: ExpandableSectionProps) {
  const [open, setOpen] = useState(false);

  return (
    <section className={`glass-card overflow-hidden transition-all duration-500 ${open ? "border-white/20 shadow-glow" : "hover:border-white/15"}`}>
      <button
        type="button"
        className="flex w-full items-center justify-between px-6 py-5 text-left transition-colors hover:bg-white/[0.02]"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
      >
        <div className="space-y-1">
          <p className="font-[var(--font-sora)] text-sm font-semibold tracking-wide text-ink">{title}</p>
          {subtitle ? <p className="text-[11px] font-medium leading-none text-dim/70 uppercase tracking-wider">{subtitle}</p> : null}
        </div>
        <div className={`flex h-8 w-8 items-center justify-center rounded-full bg-white/[0.03] border border-white/5 transition-transform duration-500 ${open ? "rotate-180 bg-white/[0.08]" : ""}`}>
          <ChevronDown className="h-4 w-4 text-ink" />
        </div>
      </button>
      <div className={`grid transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] ${open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}>
        <div className="overflow-hidden">
          <div className="border-t border-white/5 bg-white/[0.01] px-6 py-6">
            {children}
          </div>
        </div>
      </div>
    </section>
  );
}
