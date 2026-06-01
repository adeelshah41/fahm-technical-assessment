import { useState, useRef, useCallback } from "react"

const ACCEPTED_TYPES = [
  "image/png",
  "image/jpeg",
  "image/jpg",
  "application/dicom",
  "application/octet-stream",  // fallback for .dcm
]

const ACCEPTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".dcm"]

function isAcceptedFile(file) {
  const ext = "." + file.name.split(".").pop().toLowerCase()
  return ACCEPTED_TYPES.includes(file.type) || ACCEPTED_EXTENSIONS.includes(ext)
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B"
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB"
  return (bytes / (1024 * 1024)).toFixed(1) + " MB"
}

export default function UploadZone({ onFileSelect, file, preview, disabled }) {
  const [isDragging, setIsDragging] = useState(false)
  const [rejectMessage, setRejectMessage] = useState(null)
  const inputRef = useRef(null)
  const dragCounterRef = useRef(0)

  const handleFile = useCallback(
    (f) => {
      setRejectMessage(null)
      if (!isAcceptedFile(f)) {
        setRejectMessage("Unsupported file type. Please upload PNG, JPEG, or DICOM (.dcm) files.")
        return
      }
      onFileSelect(f)
    },
    [onFileSelect]
  )

  const onDragEnter = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounterRef.current += 1
    setIsDragging(true)
  }, [])

  const onDragLeave = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounterRef.current -= 1
    if (dragCounterRef.current === 0) setIsDragging(false)
  }, [])

  const onDragOver = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragging(false)
      dragCounterRef.current = 0
      if (disabled) return
      const droppedFile = e.dataTransfer.files?.[0]
      if (droppedFile) handleFile(droppedFile)
    },
    [disabled, handleFile]
  )

  const onInputChange = useCallback(
    (e) => {
      const selected = e.target.files?.[0]
      if (selected) handleFile(selected)
      e.target.value = ""
    },
    [handleFile]
  )

  /* ─── If a file has been selected, show the preview card ─── */
  if (file) {
    return (
      <div className="animate-fade-in-scale">
        <div className="glass rounded-2xl glow-teal overflow-hidden">
          {/* Preview area */}
          <div className="relative bg-slate-900/50 flex items-center justify-center min-h-[240px] sm:min-h-[320px]">
            {preview ? (
              <img
                src={preview}
                alt="Mammogram preview"
                className="max-h-[400px] w-auto object-contain p-4"
              />
            ) : (
              <div className="flex flex-col items-center gap-3 p-8 text-center">
                <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center">
                  <svg className="w-8 h-8 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
                  </svg>
                </div>
                <p className="text-sm text-slate-400">DICOM file loaded</p>
                <p className="text-xs text-slate-500">Preview not available for DICOM format</p>
              </div>
            )}

            {/* File type badge */}
            <div className="absolute top-3 right-3 glass-light rounded-lg px-2.5 py-1 text-xs text-slate-300 font-medium uppercase tracking-wider">
              {file.name.split(".").pop()}
            </div>
          </div>

          {/* File info bar */}
          <div className="px-5 py-4 border-t border-slate-700/50 flex items-center justify-between">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-8 h-8 rounded-lg bg-teal-500/10 flex items-center justify-center shrink-0">
                <svg className="w-4 h-4 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0 0 22.5 18.75V5.25A2.25 2.25 0 0 0 20.25 3H3.75A2.25 2.25 0 0 0 1.5 5.25v13.5A2.25 2.25 0 0 0 3.75 21Z" />
                </svg>
              </div>
              <div className="min-w-0">
                <p className="text-sm font-medium text-slate-200 truncate">{file.name}</p>
                <p className="text-xs text-slate-500">{formatFileSize(file.size)}</p>
              </div>
            </div>
            <div className="flex items-center gap-1.5 text-emerald-400">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
              </svg>
              <span className="text-xs font-medium">Ready</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  /* ─── Default: Drop zone ─── */
  return (
    <div className="animate-fade-in-up">
      <div
        role="button"
        tabIndex={0}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            if (!disabled) inputRef.current?.click()
          }
        }}
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
        className={`
          upload-zone glass rounded-2xl p-8 sm:p-12
          cursor-pointer select-none
          transition-all duration-300 ease-out
          ${isDragging
            ? "drag-active glow-teal-strong border-teal-500/30 bg-teal-500/[0.03]"
            : "hover:glow-teal"
          }
          ${disabled ? "opacity-50 pointer-events-none" : ""}
        `}
      >
        <div className="flex flex-col items-center text-center">
          {/* Icon */}
          <div className={`relative mb-6 transition-transform duration-300 ${isDragging ? "scale-110" : ""}`}>
            <div className={`absolute inset-0 rounded-full bg-teal-500/20 blur-xl transition-opacity duration-300 ${isDragging ? "opacity-100" : "opacity-0"}`} />
            <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-teal-500/10 to-cyan-500/10 flex items-center justify-center animate-float">
              <svg
                className={`w-9 h-9 transition-colors duration-300 ${isDragging ? "text-teal-300" : "text-teal-400/70"}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
                />
              </svg>
            </div>
          </div>

          {/* Text */}
          <h3 className="text-lg font-semibold text-slate-200 mb-1">
            {isDragging ? "Drop your image here" : "Upload Mammogram"}
          </h3>
          <p className="text-sm text-slate-400 mb-4">
            {isDragging
              ? "Release to upload"
              : "Drag and drop or click to browse"
            }
          </p>

          {/* Supported formats */}
          <div className="flex flex-wrap justify-center gap-2">
            {["PNG", "JPEG", "DICOM"].map((fmt) => (
              <span
                key={fmt}
                className="px-2.5 py-1 rounded-md bg-slate-800/60 text-xs text-slate-400 font-medium tracking-wider"
              >
                {fmt}
              </span>
            ))}
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.dcm"
          onChange={onInputChange}
          className="hidden"
          aria-label="Upload mammogram image"
        />
      </div>

      {/* Rejection message */}
      {rejectMessage && (
        <div className="mt-3 text-center animate-fade-in-up">
          <p className="text-sm text-rose-400">{rejectMessage}</p>
        </div>
      )}
    </div>
  )
}
