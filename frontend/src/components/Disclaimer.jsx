import { useState } from "react"

export default function Disclaimer() {
  const [dismissed, setDismissed] = useState(false)

  if (dismissed) return null

  return (
    <div className="animate-fade-in-up">
      <div className="relative glass-light rounded-xl px-4 py-3 flex items-start gap-3">
        {/* Icon */}
        <div className="shrink-0 mt-0.5">
          <div className="w-6 h-6 rounded-md bg-amber-500/10 flex items-center justify-center">
            <svg className="w-3.5 h-3.5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
            </svg>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-amber-400/90 uppercase tracking-wider mb-0.5">
            Medical Disclaimer
          </p>
          <p className="text-xs text-slate-400 leading-relaxed">
            This tool is for <span className="text-slate-300">research and educational purposes only</span>. 
            It is not a substitute for professional medical advice, diagnosis, or treatment. 
            Always seek the guidance of a qualified healthcare provider.
          </p>
        </div>

        {/* Dismiss button */}
        <button
          onClick={() => setDismissed(true)}
          className="shrink-0 text-slate-500 hover:text-slate-300 transition-colors p-1 -mr-1 cursor-pointer"
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
