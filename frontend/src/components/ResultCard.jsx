import { useEffect, useState } from "react"
import MetricsPanel from "./MetricsPanel"

/* ─── Helper: map label to clinical theme ─── */
function getLabelConfig(label) {
  const normalized = (label || "").toLowerCase()

  if (normalized.includes("malignant")) {
    return {
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
        </svg>
      ),
      label: "Malignant",
      subtitle: "Abnormality Detected — Refer for Biopsy",
      riskLevel: "HIGH",
      textClass: "text-rose-400",
      bgClass: "bg-rose-500/8",
      borderClass: "border-rose-500/20",
      barGradient: "from-rose-500 to-rose-400",
      glowClass: "glow-rose",
      gaugeColor: "#f87171",
      riskBadgeBg: "bg-rose-500/15",
      riskBadgeText: "text-rose-400",
      riskBadgeBorder: "border-rose-500/25",
    }
  }

  if (normalized.includes("benign")) {
    return {
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
        </svg>
      ),
      label: "Benign",
      subtitle: "Non-Malignant Finding — Routine Follow-up",
      riskLevel: "LOW",
      textClass: "text-emerald-400",
      bgClass: "bg-emerald-500/8",
      borderClass: "border-emerald-500/20",
      barGradient: "from-emerald-500 to-teal-400",
      glowClass: "glow-emerald",
      gaugeColor: "#34d399",
      riskBadgeBg: "bg-emerald-500/15",
      riskBadgeText: "text-emerald-400",
      riskBadgeBorder: "border-emerald-500/25",
    }
  }

  return {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
      </svg>
    ),
    label: "Normal",
    subtitle: "No Abnormalities Detected",
    riskLevel: "NONE",
    textClass: "text-emerald-400",
    bgClass: "bg-emerald-500/8",
    borderClass: "border-emerald-500/20",
    barGradient: "from-emerald-500 to-teal-400",
    glowClass: "glow-emerald",
    gaugeColor: "#34d399",
    riskBadgeBg: "bg-emerald-500/15",
    riskBadgeText: "text-emerald-400",
    riskBadgeBorder: "border-emerald-500/25",
  }
}

/* ─── Confidence Gauge (Radial) ─── */
function ConfidenceGauge({ confidence, config }) {
  const [offset, setOffset] = useState(251.2)
  const percentage = Math.round((confidence ?? 0) * 100)
  const circumference = 2 * Math.PI * 40 // r=40
  const targetOffset = circumference - (circumference * percentage) / 100

  useEffect(() => {
    const timer = setTimeout(() => setOffset(targetOffset), 150)
    return () => clearTimeout(timer)
  }, [targetOffset])

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-28 h-28">
        <svg className="gauge-ring w-full h-full" viewBox="0 0 100 100">
          <circle className="gauge-track" cx="50" cy="50" r="40" />
          <circle
            className="gauge-fill"
            cx="50"
            cy="50"
            r="40"
            stroke={config.gaugeColor}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-2xl font-bold font-mono tabular-nums ${config.textClass}`}>
            {percentage}
          </span>
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-semibold">
            percent
          </span>
        </div>
      </div>
      <p className="text-[10px] text-slate-500 uppercase tracking-widest mt-2 font-medium">Confidence</p>
    </div>
  )
}

/* ─── Confidence Bar (Linear) ─── */
function ConfidenceBar({ confidence, config }) {
  const [width, setWidth] = useState(0)
  const percentage = Math.round((confidence ?? 0) * 100)

  useEffect(() => {
    const timer = setTimeout(() => setWidth(percentage), 100)
    return () => clearTimeout(timer)
  }, [percentage])

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Confidence Score</span>
        <span className={`text-xl font-bold tabular-nums font-mono ${config.textClass}`}>
          {percentage}
          <span className="text-xs font-medium text-slate-500 ml-0.5">%</span>
        </span>
      </div>
      <div className="h-2.5 rounded-full bg-slate-800/80 overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${config.barGradient} confidence-bar-fill transition-all duration-1000 ease-out`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════
   ResultCard — Multi-Tab Clinical Dashboard
   ═══════════════════════════════════════════ */
export default function ResultCard({ result, onReset }) {
  const [activeTab, setActiveTab] = useState("diagnosis")

  const label = result?.prediction || result?.label || "Unknown"
  const confidence = result?.confidence ?? result?.probability ?? 0
  const config = getLabelConfig(label)

  const tabs = [
    { id: "diagnosis", label: "Diagnosis" },
    { id: "performance", label: "Model Metrics" },
  ]

  return (
    <>
      <div className="animate-fade-in-up" style={{ animationDelay: "0.1s", opacity: 0 }}>
        <div className={`glass rounded-2xl overflow-hidden border ${config.borderClass} shadow-lg ${config.glowClass}`}>

          {/* ── Header: Diagnosis Summary ── */}
          <div className={`relative ${config.bgClass} px-5 py-4 sm:px-6 sm:py-5`}>
            <div className="flex items-center gap-4">
              {/* Icon */}
              <div className={`
                w-12 h-12 rounded-xl
                ${config.bgClass} border ${config.borderClass}
                flex items-center justify-center
                ${config.textClass}
              `}>
                {config.icon}
              </div>

              {/* Label & subtitle */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2.5 mb-0.5">
                  <h2 className={`text-2xl font-bold ${config.textClass}`}>
                    {config.label}
                  </h2>
                  <span className={`
                    px-2 py-0.5 rounded-md text-[9px] font-bold uppercase tracking-widest
                    ${config.riskBadgeBg} ${config.riskBadgeText} border ${config.riskBadgeBorder}
                  `}>
                    {config.riskLevel} Risk
                  </span>
                </div>
                <p className="text-xs text-slate-500">{config.subtitle}</p>
              </div>
            </div>

            <div className={`absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent ${config.borderClass} to-transparent`} />
          </div>

          {/* ── Tab Navigation ── */}
          <div className="flex border-b border-slate-700/30 px-5 sm:px-6">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`tab-button ${activeTab === tab.id ? "active" : ""}`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* ── Tab Content ── */}
          <div className="px-5 py-5 sm:px-6">

            {/* Tab 1: Diagnosis */}
            {activeTab === "diagnosis" && (
              <div className="space-y-5 tab-content">
                {/* Gauge + Bar */}
                <div className="flex flex-col sm:flex-row items-center gap-6">
                  <ConfidenceGauge confidence={confidence} config={config} />
                  <div className="flex-1 w-full space-y-4">
                    <ConfidenceBar confidence={confidence} config={config} />

                    {/* Raw label */}
                    <div className="flex items-center gap-3 p-3 rounded-xl glass-inset">
                      <svg className="w-4 h-4 text-slate-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9.568 3H5.25A2.25 2.25 0 0 0 3 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 0 0 5.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 0 0 9.568 3Z" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 6h.008v.008H6V6Z" />
                      </svg>
                      <div className="min-w-0">
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider">Raw Prediction</p>
                        <p className="text-sm text-slate-300 font-mono truncate">{label}</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Clinical disclaimer inside result */}
                <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-amber-500/[0.04] border border-amber-500/10">
                  <svg className="w-4 h-4 text-amber-500/70 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z" />
                  </svg>
                  <p className="text-[11px] text-amber-500/60 leading-relaxed">
                    This AI prediction is for research and educational purposes only.
                    Always consult a qualified radiologist for clinical diagnosis.
                  </p>
                </div>
              </div>
            )}

            {/* Tab 2: Model Performance Metrics */}
            {activeTab === "performance" && <MetricsPanel />}
          </div>

          {/* ── Footer: Actions ── */}
          <div className="px-5 py-4 sm:px-6 border-t border-slate-700/30 flex justify-end">
            <button
              onClick={onReset}
              className="btn-primary text-xs sm:text-sm"
              id="analyze-another-btn"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
              </svg>
              Analyze Another Image
            </button>
          </div>
        </div>
      </div>
    </>
  )
}
