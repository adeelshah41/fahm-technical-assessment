import { useEffect, useState } from "react"

/* ─── Helper: map label to theme config ─── */
function getLabelConfig(label) {
  const normalized = (label || "").toLowerCase()

  if (normalized.includes("malignant")) {
    return {
      color: "rose",
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
        </svg>
      ),
      label: "Malignant",
      subtitle: "Abnormality detected",
      bgClass: "from-rose-500/10 to-rose-600/5",
      borderClass: "border-rose-500/20",
      barClass: "from-rose-500 to-rose-400",
      textClass: "text-rose-400",
      glowClass: "shadow-rose-500/20",
    }
  }

  if (normalized.includes("benign")) {
    return {
      color: "amber",
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
        </svg>
      ),
      label: "Benign",
      subtitle: "Non-malignant finding",
      bgClass: "from-amber-500/10 to-amber-600/5",
      borderClass: "border-amber-500/20",
      barClass: "from-amber-500 to-yellow-400",
      textClass: "text-amber-400",
      glowClass: "shadow-amber-500/20",
    }
  }

  // Normal / Negative
  return {
    color: "emerald",
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
      </svg>
    ),
    label: "Normal",
    subtitle: "No abnormalities detected",
    bgClass: "from-emerald-500/10 to-emerald-600/5",
    borderClass: "border-emerald-500/20",
    barClass: "from-emerald-500 to-teal-400",
    textClass: "text-emerald-400",
    glowClass: "shadow-emerald-500/20",
  }
}

/* ─── Confidence Bar Component ─── */
function ConfidenceBar({ confidence, config }) {
  const [width, setWidth] = useState(0)
  const percentage = Math.round((confidence ?? 0) * 100)

  useEffect(() => {
    // Trigger animation after mount
    const timer = setTimeout(() => setWidth(percentage), 100)
    return () => clearTimeout(timer)
  }, [percentage])

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Confidence</span>
        <span className={`text-2xl font-bold tabular-nums ${config.textClass}`}>
          {percentage}
          <span className="text-sm font-medium text-slate-500 ml-0.5">%</span>
        </span>
      </div>
      <div className="h-3 rounded-full bg-slate-800/80 overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${config.barClass} confidence-bar-fill transition-all duration-1000 ease-out`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════
   ResultCard Component
   ═══════════════════════════════════════════ */
export default function ResultCard({ result, onReset }) {
  const label = result?.prediction || result?.label || "Unknown"
  const confidence = result?.confidence ?? result?.probability ?? 0
  const config = getLabelConfig(label)

  return (
    <div className="animate-fade-in-up" style={{ animationDelay: "0.1s", opacity: 0 }}>
      <div
        className={`
          glass rounded-2xl overflow-hidden
          border ${config.borderClass}
          shadow-lg ${config.glowClass}
        `}
      >
        {/* Header */}
        <div className={`relative bg-gradient-to-r ${config.bgClass} px-6 py-5`}>
          <div className="flex items-center gap-4">
            <div className={`
              w-12 h-12 rounded-xl
              bg-gradient-to-br ${config.bgClass}
              border ${config.borderClass}
              flex items-center justify-center
              ${config.textClass}
            `}>
              {config.icon}
            </div>
            <div>
              <p className="text-xs font-medium text-slate-400 uppercase tracking-widest mb-0.5">
                Classification Result
              </p>
              <h2 className={`text-2xl font-bold ${config.textClass}`}>
                {config.label}
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">{config.subtitle}</p>
            </div>
          </div>

          {/* Decorative gradient line */}
          <div className={`absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent ${config.borderClass} to-transparent`} />
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-5">
          {/* Confidence bar */}
          <ConfidenceBar confidence={confidence} config={config} />

          {/* Raw prediction label */}
          <div className="flex items-center gap-3 p-3 rounded-xl bg-slate-800/40">
            <svg className="w-4 h-4 text-slate-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.568 3H5.25A2.25 2.25 0 0 0 3 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 0 0 5.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 0 0 9.568 3Z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 6h.008v.008H6V6Z" />
            </svg>
            <div className="min-w-0">
              <p className="text-xs text-slate-500">Raw Prediction</p>
              <p className="text-sm text-slate-300 font-mono truncate">{label}</p>
            </div>
          </div>

          {/* Disclaimer inside result */}
          <div className="flex items-start gap-2.5 p-3 rounded-xl bg-amber-500/[0.04] border border-amber-500/10">
            <svg className="w-4 h-4 text-amber-500/70 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z" />
            </svg>
            <p className="text-xs text-amber-500/60 leading-relaxed">
              This AI prediction is for research and educational purposes only. 
              Always consult a qualified radiologist for clinical diagnosis.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-700/30 flex justify-end">
          <button
            onClick={onReset}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl
                       bg-slate-800/60 text-slate-300 text-sm font-medium
                       hover:bg-slate-700/60 hover:text-white
                       active:scale-[0.98]
                       transition-all duration-200 cursor-pointer"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
            </svg>
            Analyze Another
          </button>
        </div>
      </div>
    </div>
  )
}
