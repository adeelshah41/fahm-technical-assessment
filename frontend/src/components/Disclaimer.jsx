import { useState } from "react"

export default function Disclaimer() {
  const [dismissed, setDismissed] = useState(false)

  if (dismissed) return null

  return (
    <div className="animate-fade-in-up">
      <div className="relative glass-light rounded-xl px-4 py-3 flex items-start gap-3 border-l-2 border-amber-500/40">
        {/* Shield Icon */}
        <div className="shrink-0 mt-0.5">
          <div className="w-7 h-7 rounded-lg bg-amber-500/10 flex items-center justify-center">
            <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m0-10.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z" />
            </svg>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <p className="text-[11px] font-semibold text-amber-400/90 uppercase tracking-widest mb-0.5">
            Clinical Disclaimer
          </p>
          <p className="text-xs text-slate-400 leading-relaxed">
            This AI system is for <span className="text-slate-300 font-medium">research and educational purposes only</span>.
            It is not intended as a substitute for professional medical diagnosis.
            Always consult a board-certified radiologist for clinical evaluation.
          </p>
        </div>

        {/* Dismiss */}
        <button
          onClick={() => setDismissed(true)}
          className="shrink-0 text-slate-600 hover:text-slate-400 transition-colors p-1 -mr-1 cursor-pointer"
          aria-label="Dismiss disclaimer"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  )
}
