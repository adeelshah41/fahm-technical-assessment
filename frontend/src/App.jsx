import { useState, useCallback, useEffect } from "react"
import UploadZone from "./components/UploadZone"
import ResultCard from "./components/ResultCard"
import Disclaimer from "./components/Disclaimer"
import { predictImage } from "./services/api"

/* ─── Animated Background ─── */
function BackgroundEffects() {
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
      <div className="absolute -top-60 -left-60 w-[600px] h-[600px] rounded-full bg-teal-500/[0.03] blur-[120px]" />
      <div className="absolute top-1/4 -right-40 w-[500px] h-[500px] rounded-full bg-cyan-500/[0.025] blur-[100px]" />
      <div className="absolute -bottom-40 left-1/4 w-[500px] h-[500px] rounded-full bg-teal-400/[0.02] blur-[100px]" />
    </div>
  )
}

/* ─── Header with System Status ─── */
function Header({ apiStatus }) {
  return (
    <header className="relative z-10 pt-6 pb-2">
      {/* Top bar with system status */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <div className={`status-dot ${apiStatus === "online" ? "status-dot-online" : "status-dot-offline"}`} />
          <span className="text-[10px] font-medium text-slate-500 uppercase tracking-widest">
            {apiStatus === "online" ? "System Online" : apiStatus === "checking" ? "Connecting..." : "System Offline"}
          </span>
        </div>
        <span className="text-[10px] text-slate-600 font-mono">v1.0.0</span>
      </div>

      {/* Logo & Title */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center mb-5">
          <div className="relative">
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-teal-500 to-cyan-500 blur-xl opacity-20 pulse-ring" />
            <div className="relative glass-raised rounded-2xl p-3.5 glow-teal border border-teal-500/10">
              <svg className="w-9 h-9 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
              </svg>
            </div>
          </div>
        </div>

        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight">
          <span className="bg-gradient-to-r from-teal-400 via-cyan-400 to-teal-300 bg-clip-text text-transparent">
            FAHM
          </span>
        </h1>
        <p className="mt-2 text-base sm:text-lg text-slate-400 font-light tracking-wide">
          AI-Powered Mammogram Classification
        </p>
        <div className="mt-2 flex items-center justify-center gap-3">
          <span className="text-[10px] text-slate-600 uppercase tracking-[0.2em] font-medium">Deep Learning</span>
          <span className="w-1 h-1 rounded-full bg-slate-700" />
          <span className="text-[10px] text-slate-600 uppercase tracking-[0.2em] font-medium">Medical Imaging</span>
          <span className="w-1 h-1 rounded-full bg-slate-700" />
          <span className="text-[10px] text-slate-600 uppercase tracking-[0.2em] font-medium">Early Detection</span>
        </div>
      </div>
    </header>
  )
}

/* ─── Loading Overlay with Radar Animation ─── */
function LoadingOverlay() {
  return (
    <div className="animate-fade-in-up mt-8">
      <div className="glass rounded-2xl p-8 glow-teal text-center max-w-md mx-auto">
        {/* Radar spinner */}
        <div className="relative w-20 h-20 mx-auto mb-6">
          {/* Outer ring */}
          <div className="absolute inset-0 rounded-full border border-slate-700/50" />
          <div className="absolute inset-2 rounded-full border border-slate-700/30" />
          <div className="absolute inset-4 rounded-full border border-slate-700/20" />

          {/* Radar sweep */}
          <div className="absolute inset-0 animate-radar">
            <div
              className="absolute top-1/2 left-1/2 w-1/2 h-0.5 origin-left"
              style={{
                background: "linear-gradient(90deg, rgba(20, 184, 166, 0.8), transparent)",
                transform: "translateY(-50%)",
              }}
            />
          </div>

          {/* Center dot */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-2.5 h-2.5 rounded-full bg-teal-400 pulse-ring" />
          </div>
        </div>

        <h3 className="text-lg font-semibold text-slate-200 mb-1.5">
          Analyzing Mammogram
        </h3>
        <p className="text-sm text-slate-500 mb-5">
          Running EfficientNet-B0 inference…
        </p>

        {/* Dot loader */}
        <div className="dot-loader flex justify-center gap-2">
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
      <div className="rounded-xl border border-rose-500/20 bg-rose-500/[0.06] backdrop-blur-sm p-4 flex items-start gap-3">
        <div className="shrink-0 mt-0.5">
          <div className="w-7 h-7 rounded-lg bg-rose-500/10 flex items-center justify-center">
            <svg className="w-4 h-4 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
            </svg>
          </div>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-rose-300">Analysis Failed</p>
          <p className="text-xs text-rose-400/70 mt-0.5 leading-relaxed">{message}</p>
        </div>
        <button
          onClick={onDismiss}
          className="shrink-0 text-rose-400/50 hover:text-rose-300 transition-colors cursor-pointer p-1"
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
  const [apiStatus, setApiStatus] = useState("checking")

  // Check API health on mount
  useEffect(() => {
    const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080"
    fetch(`${API_URL}/health`, { mode: "cors" })
      .then((res) => {
        if (res.ok) setApiStatus("online")
        else setApiStatus("offline")
      })
      .catch(() => setApiStatus("offline"))
  }, [])

  const handleFileSelect = useCallback((selectedFile) => {
    setFile(selectedFile)
    setResult(null)
    setError(null)

    if (selectedFile.type.startsWith("image/")) {
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(selectedFile)
    } else {
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
    <div className="min-h-screen bg-clinical bg-grid relative">
      <BackgroundEffects />

      <div className="relative z-10 max-w-3xl mx-auto px-4 sm:px-6 pb-16">
        <Header apiStatus={apiStatus} />

        {/* Disclaimer */}
        <div className="mt-5">
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
              id="analyze-btn"
              onClick={handleAnalyze}
              className="btn-primary"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
              </svg>
              Analyze Mammogram
            </button>
            <button
              onClick={handleReset}
              className="btn-secondary"
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
          <div className="h-px bg-gradient-to-r from-transparent via-slate-800/60 to-transparent mb-6" />
          <div className="flex items-center justify-center gap-3 text-[10px] text-slate-600">
            <span className="uppercase tracking-widest font-medium">FAHM</span>
            <span className="w-1 h-1 rounded-full bg-slate-700" />
            <span>EfficientNet-B0 · CBIS-DDSM</span>
            <span className="w-1 h-1 rounded-full bg-slate-700" />
            <span>Built for early breast cancer detection</span>
          </div>
        </footer>
      </div>
    </div>
  )
}
