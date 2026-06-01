import { useState, useCallback } from "react"
import UploadZone from "./components/UploadZone"
import ResultCard from "./components/ResultCard"
import Disclaimer from "./components/Disclaimer"
import { predictImage } from "./services/api"

/* ─── Animated background orbs ─── */
function BackgroundOrbs() {
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
      <div className="absolute -top-40 -left-40 w-[500px] h-[500px] rounded-full bg-teal-500/[0.04] blur-[100px]" />
      <div className="absolute top-1/3 -right-32 w-[400px] h-[400px] rounded-full bg-cyan-500/[0.04] blur-[100px]" />
      <div className="absolute -bottom-40 left-1/3 w-[450px] h-[450px] rounded-full bg-teal-400/[0.03] blur-[100px]" />
    </div>
  )
}

/* ─── Header with logo ─── */
function Header() {
  return (
    <header className="relative z-10 pt-8 pb-4 text-center">
      {/* Logo mark */}
      <div className="inline-flex items-center justify-center mb-6">
        <div className="relative">
          <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-teal-500 to-cyan-500 blur-xl opacity-30 pulse-ring" />
          <div className="relative glass rounded-2xl p-4 glow-teal">
            <svg className="w-10 h-10 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
            </svg>
          </div>
        </div>
      </div>

      {/* Title */}
      <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
        <span className="bg-gradient-to-r from-teal-400 via-cyan-400 to-teal-300 bg-clip-text text-transparent">
          FAHM
        </span>
      </h1>
      <p className="mt-2 text-base sm:text-lg text-slate-400 font-light tracking-wide">
        AI-Powered Mammogram Classification
      </p>
      <p className="mt-1 text-xs text-slate-500 tracking-widest uppercase">
        Deep Learning &middot; Medical Imaging &middot; Early Detection
      </p>
    </header>
  )
}

/* ─── Loading Overlay ─── */
function LoadingOverlay() {
  return (
    <div className="animate-fade-in-up mt-8">
      <div className="glass rounded-2xl p-8 glow-teal text-center max-w-md mx-auto">
        {/* Spinner */}
        <div className="relative w-16 h-16 mx-auto mb-6">
          <div className="absolute inset-0 rounded-full border-2 border-slate-700" />
          <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-teal-400 animate-spin-slow" />
          <div className="absolute inset-2 rounded-full border-2 border-transparent border-b-cyan-400 animate-spin-slow" style={{ animationDirection: "reverse", animationDuration: "2s" }} />
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-teal-400 pulse-ring" />
          </div>
        </div>

        <h3 className="text-lg font-semibold text-slate-200 mb-2">
          Analyzing Mammogram
        </h3>
        <p className="text-sm text-slate-400 mb-4">
          Running deep learning inference…
        </p>

        {/* Dot loader */}
        <div className="dot-loader flex justify-center gap-1.5">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="inline-block w-2 h-2 rounded-full bg-teal-400"
              style={{ animation: "dot-bounce 1.4s infinite ease-in-out both" }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

/* ─── Error Banner ─── */
function ErrorBanner({ message, onDismiss }) {
  return (
    <div className="animate-fade-in-up mt-6 max-w-lg mx-auto">
      <div className="rounded-xl border border-rose-500/20 bg-rose-500/[0.08] backdrop-blur-sm p-4 flex items-start gap-3">
        <div className="shrink-0 mt-0.5">
          <svg className="w-5 h-5 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-rose-300">Analysis Failed</p>
          <p className="text-sm text-rose-400/80 mt-0.5">{message}</p>
        </div>
        <button
          onClick={onDismiss}
          className="shrink-0 text-rose-400/60 hover:text-rose-300 transition-colors cursor-pointer"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════
   Main App Component
   ═══════════════════════════════════════════ */
export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileSelect = useCallback((selectedFile) => {
    setFile(selectedFile)
    setResult(null)
    setError(null)

    // Generate preview for image files
    if (selectedFile.type.startsWith("image/")) {
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(selectedFile)
    } else {
      // DICOM or other — show placeholder
      setPreview(null)
    }
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!file) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await predictImage(file)
      setResult(data)
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        "An unexpected error occurred. Please try again."
      setError(message)
    } finally {
      setLoading(false)
    }
  }, [file])

  const handleReset = useCallback(() => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }, [])

  return (
    <div className="min-h-screen bg-mesh relative">
      <BackgroundOrbs />

      <div className="relative z-10 max-w-3xl mx-auto px-4 sm:px-6 pb-16">
        <Header />

        {/* Disclaimer */}
        <div className="mt-6">
          <Disclaimer />
        </div>

        {/* Upload Zone */}
        <div className="mt-8">
          <UploadZone
            onFileSelect={handleFileSelect}
            file={file}
            preview={preview}
            disabled={loading}
          />
        </div>

        {/* Analyze Button */}
        {file && !result && !loading && (
          <div className="animate-fade-in-up mt-6 flex justify-center gap-3">
            <button
              onClick={handleAnalyze}
              className="group relative inline-flex items-center gap-2.5 px-8 py-3.5 rounded-xl
                         bg-gradient-to-r from-teal-500 to-cyan-500
                         text-white font-semibold text-sm tracking-wide
                         shadow-lg shadow-teal-500/25
                         hover:shadow-xl hover:shadow-teal-500/30
                         hover:from-teal-400 hover:to-cyan-400
                         active:scale-[0.98]
                         transition-all duration-200 cursor-pointer"
            >
              <svg className="w-5 h-5 transition-transform group-hover:scale-110" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
              </svg>
              Analyze Mammogram
              <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-teal-400/0 via-white/10 to-cyan-400/0
                              translate-x-[-100%] group-hover:translate-x-[100%]
                              transition-transform duration-700" />
            </button>
            <button
              onClick={handleReset}
              className="inline-flex items-center gap-2 px-5 py-3.5 rounded-xl
                         glass text-slate-400 text-sm font-medium
                         hover:text-slate-200 hover:border-slate-600
                         active:scale-[0.98]
                         transition-all duration-200 cursor-pointer"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
              </svg>
              Clear
            </button>
          </div>
        )}

        {/* Loading */}
        {loading && <LoadingOverlay />}

        {/* Error */}
        {error && (
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        )}

        {/* Result */}
        {result && (
          <div className="mt-8">
            <ResultCard result={result} onReset={handleReset} />
          </div>
        )}

        {/* Footer */}
        <footer className="mt-16 text-center">
          <div className="h-px bg-gradient-to-r from-transparent via-slate-700/50 to-transparent mb-6" />
          <p className="text-xs text-slate-600">
            FAHM &middot; Built with care for early breast cancer detection
          </p>
        </footer>
      </div>
    </div>
  )
}
