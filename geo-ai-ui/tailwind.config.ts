import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        surface: "#0a0c10",
        panel: "rgba(18, 22, 30, 0.7)",
        panelSoft: "rgba(26, 34, 48, 0.5)",
        borderSoft: "rgba(255, 255, 255, 0.06)",
        ink: "#f8fafc",
        dim: "#94a3b8",
        brand: {
          50: "#eff6ff",
          100: "#dbeafe",
          200: "#bfdbfe",
          300: "#93c5fd",
          400: "#60a5fa",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          800: "#1e40af",
          900: "#1e3a8a",
          950: "#172554",
        },
        low: "#10b981",
        moderate: "#f59e0b",
        high: "#ef4444"
      },
      boxShadow: {
        soft: "0 20px 50px rgba(0, 0, 0, 0.35)",
        card: "0 8px 30px rgba(0, 0, 0, 0.3)",
        glow: "0 0 20px rgba(59, 130, 246, 0.15)",
        "glow-lg": "0 0 40px rgba(59, 130, 246, 0.25)",
      },
      borderRadius: {
        xl2: "1.25rem",
        xl3: "1.75rem",
      },
      backgroundImage: {
        "glass-gradient": "linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 100%)",
        "mesh-gradient": "radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0, transparent 50%), radial-gradient(at 50% 0%, rgba(16, 185, 129, 0.1) 0, transparent 50%), radial-gradient(at 100% 0%, rgba(245, 158, 11, 0.05) 0, transparent 50%)",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        pulseSoft: {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" }
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" }
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-6px)" }
        }
      },
      animation: {
        fadeUp: "fadeUp 500ms cubic-bezier(0.16, 1, 0.3, 1)",
        pulseSoft: "pulseSoft 2s ease-in-out infinite",
        shimmer: "shimmer 2s infinite",
        float: "float 4s ease-in-out infinite",
      }
    }
  },
  plugins: []
};

export default config;
