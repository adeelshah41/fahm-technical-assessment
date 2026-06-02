import { useState, useRef } from "react"

/* ═══════════════════════════════════════════
   ReportModal — Printable Clinical Report
   Generates a clean, medical-style PDF report
   that doctors can save or print.
   ═══════════════════════════════════════════ */

function getCurrentTimestamp() {
  const now = new Date()
  return now.toLocaleString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  })
}

export default function ReportModal({ result, onClose }) {
  const [patientName, setPatientName] = useState("")
  const [patientId, setPatientId] = useState("")
  const [patientAge, setPatientAge] = useState("")
  const [notes, setNotes] = useState("")
  const reportRef = useRef(null)

  const label = result?.prediction || result?.label || "Unknown"
  const confidence = result?.confidence ?? result?.probability ?? 0
  const confidencePct = Math.round(confidence * 100)
  const isMalignant = label.toLowerCase().includes("malignant")
  const timestamp = getCurrentTimestamp()

  const handlePrint = () => {
    window.print()
  }

  return (
    <div className="report-modal-overlay no-print" onClick={onClose}>
      <div
        className="report-content"
        onClick={(e) => e.stopPropagation()}
      >
        <div ref={reportRef}>
          {/* Report Header */}
          <div style={{
            background: "linear-gradient(135deg, #0d9488, #0891b2)",
            padding: "2rem 2.5rem",
            color: "white",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: 0, letterSpacing: "-0.02em" }}>
                  🔬 FAHM Diagnostic Report
                </h1>
                <p style={{ fontSize: "0.8125rem", opacity: 0.85, marginTop: "0.25rem" }}>
                  AI-Assisted Mammogram Classification Report
                </p>
              </div>
              <div style={{ textAlign: "right", fontSize: "0.75rem", opacity: 0.8 }}>
                <p style={{ margin: 0 }}>Report Generated</p>
                <p style={{ margin: 0, fontWeight: 600 }}>{timestamp}</p>
              </div>
            </div>
          </div>

          {/* Body */}
          <div style={{ padding: "2rem 2.5rem" }}>
            {/* Patient Information */}
            <section style={{ marginBottom: "1.5rem" }}>
              <h2 style={{
                fontSize: "0.75rem",
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                color: "#64748b",
                borderBottom: "2px solid #e2e8f0",
                paddingBottom: "0.5rem",
                marginBottom: "1rem",
              }}>
                Patient Information
              </h2>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem 2rem" }}>
                <div>
                  <p style={{ fontSize: "0.6875rem", color: "#94a3b8", margin: "0 0 2px 0", textTransform: "uppercase", letterSpacing: "0.05em" }}>Patient Name</p>
                  <p style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#1e293b", margin: 0 }}>{patientName || "—"}</p>
                </div>
                <div>
                  <p style={{ fontSize: "0.6875rem", color: "#94a3b8", margin: "0 0 2px 0", textTransform: "uppercase", letterSpacing: "0.05em" }}>Patient ID</p>
                  <p style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#1e293b", margin: 0, fontFamily: "monospace" }}>{patientId || "—"}</p>
                </div>
                <div>
                  <p style={{ fontSize: "0.6875rem", color: "#94a3b8", margin: "0 0 2px 0", textTransform: "uppercase", letterSpacing: "0.05em" }}>Age</p>
                  <p style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#1e293b", margin: 0 }}>{patientAge ? `${patientAge} years` : "—"}</p>
                </div>
                <div>
                  <p style={{ fontSize: "0.6875rem", color: "#94a3b8", margin: "0 0 2px 0", textTransform: "uppercase", letterSpacing: "0.05em" }}>Modality</p>
                  <p style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#1e293b", margin: 0 }}>Mammography (AI-Assisted)</p>
                </div>
              </div>
            </section>

            {/* Classification Result */}
            <section style={{ marginBottom: "1.5rem" }}>
              <h2 style={{
                fontSize: "0.75rem",
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                color: "#64748b",
                borderBottom: "2px solid #e2e8f0",
                paddingBottom: "0.5rem",
                marginBottom: "1rem",
              }}>
                AI Classification Result
              </h2>

              <div style={{
                display: "flex",
                alignItems: "center",
                gap: "1.5rem",
                padding: "1.25rem",
                borderRadius: "0.75rem",
                background: isMalignant ? "#fef2f2" : "#f0fdf4",
                border: `2px solid ${isMalignant ? "#fecaca" : "#bbf7d0"}`,
              }}>
                <div style={{
                  width: 56,
                  height: 56,
                  borderRadius: "50%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "1.5rem",
                  background: isMalignant ? "#fee2e2" : "#dcfce7",
                  flexShrink: 0,
                }}>
                  {isMalignant ? "⚠️" : "✅"}
                </div>
                <div>
                  <p style={{
                    fontSize: "1.5rem",
                    fontWeight: 800,
                    color: isMalignant ? "#dc2626" : "#16a34a",
                    margin: 0,
                    letterSpacing: "-0.02em",
                  }}>
                    {label}
                  </p>
                  <p style={{ fontSize: "0.875rem", color: "#64748b", margin: "0.25rem 0 0 0" }}>
                    Model Confidence: <strong style={{ fontFamily: "monospace" }}>{confidencePct}%</strong>
                  </p>
                </div>
              </div>
            </section>

            {/* Model Information */}
            <section style={{ marginBottom: "1.5rem" }}>
              <h2 style={{
                fontSize: "0.75rem",
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                color: "#64748b",
                borderBottom: "2px solid #e2e8f0",
                paddingBottom: "0.5rem",
                marginBottom: "1rem",
              }}>
                Model Information
              </h2>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem 2rem", fontSize: "0.8125rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "0.375rem 0", borderBottom: "1px solid #f1f5f9" }}>
                  <span style={{ color: "#94a3b8" }}>Architecture</span>
                  <span style={{ fontWeight: 600, color: "#334155" }}>EfficientNet-B0</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "0.375rem 0", borderBottom: "1px solid #f1f5f9" }}>
                  <span style={{ color: "#94a3b8" }}>Training Dataset</span>
                  <span style={{ fontWeight: 600, color: "#334155" }}>CBIS-DDSM</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "0.375rem 0", borderBottom: "1px solid #f1f5f9" }}>
                  <span style={{ color: "#94a3b8" }}>AUC-ROC</span>
                  <span style={{ fontWeight: 600, color: "#334155", fontFamily: "monospace" }}>0.8417</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "0.375rem 0", borderBottom: "1px solid #f1f5f9" }}>
                  <span style={{ color: "#94a3b8" }}>Sensitivity</span>
                  <span style={{ fontWeight: 600, color: "#334155", fontFamily: "monospace" }}>74.77%</span>
                </div>
              </div>
            </section>

            {/* Clinical Notes */}
            {notes && (
              <section style={{ marginBottom: "1.5rem" }}>
                <h2 style={{
                  fontSize: "0.75rem",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                  color: "#64748b",
                  borderBottom: "2px solid #e2e8f0",
                  paddingBottom: "0.5rem",
                  marginBottom: "1rem",
                }}>
                  Clinician Notes
                </h2>
                <p style={{ fontSize: "0.875rem", color: "#334155", lineHeight: 1.7, margin: 0, whiteSpace: "pre-wrap" }}>
                  {notes}
                </p>
              </section>
            )}

            {/* Disclaimer */}
            <div style={{
              padding: "1rem",
              borderRadius: "0.5rem",
              background: "#fffbeb",
              border: "1px solid #fde68a",
              fontSize: "0.75rem",
              color: "#92400e",
              lineHeight: 1.6,
            }}>
              <strong>⚠️ Important Disclaimer:</strong> This AI-generated report is for research and educational purposes only.
              It is not a substitute for a certified radiologist's diagnosis. The model's prediction should be used as a
              supplementary screening tool and must not be used as the sole basis for clinical decision-making.
            </div>
          </div>
        </div>

        {/* Action Buttons (not printed) */}
        <div className="no-print" style={{
          padding: "1rem 2.5rem 1.5rem",
          borderTop: "1px solid #e2e8f0",
          background: "#f8fafc",
          borderRadius: "0 0 1rem 1rem",
        }}>
          {/* Input Fields */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 80px", gap: "0.75rem", marginBottom: "1rem" }}>
            <input
              type="text"
              placeholder="Patient Name"
              value={patientName}
              onChange={(e) => setPatientName(e.target.value)}
              style={{
                padding: "0.5rem 0.75rem",
                borderRadius: "0.5rem",
                border: "1px solid #e2e8f0",
                fontSize: "0.8125rem",
                outline: "none",
                color: "#1e293b",
              }}
            />
            <input
              type="text"
              placeholder="Patient ID"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              style={{
                padding: "0.5rem 0.75rem",
                borderRadius: "0.5rem",
                border: "1px solid #e2e8f0",
                fontSize: "0.8125rem",
                outline: "none",
                color: "#1e293b",
                fontFamily: "monospace",
              }}
            />
            <input
              type="number"
              placeholder="Age"
              value={patientAge}
              onChange={(e) => setPatientAge(e.target.value)}
              style={{
                padding: "0.5rem 0.75rem",
                borderRadius: "0.5rem",
                border: "1px solid #e2e8f0",
                fontSize: "0.8125rem",
                outline: "none",
                color: "#1e293b",
              }}
            />
          </div>
          <textarea
            placeholder="Add clinician notes (optional)..."
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={2}
            style={{
              width: "100%",
              padding: "0.5rem 0.75rem",
              borderRadius: "0.5rem",
              border: "1px solid #e2e8f0",
              fontSize: "0.8125rem",
              outline: "none",
              resize: "vertical",
              marginBottom: "1rem",
              fontFamily: "inherit",
              color: "#1e293b",
            }}
          />
          <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem" }}>
            <button
              onClick={onClose}
              style={{
                padding: "0.625rem 1.25rem",
                borderRadius: "0.5rem",
                border: "1px solid #e2e8f0",
                background: "white",
                color: "#64748b",
                fontSize: "0.8125rem",
                fontWeight: 500,
                cursor: "pointer",
              }}
            >
              Close
            </button>
            <button
              onClick={handlePrint}
              style={{
                padding: "0.625rem 1.5rem",
                borderRadius: "0.5rem",
                border: "none",
                background: "linear-gradient(135deg, #0d9488, #0891b2)",
                color: "white",
                fontSize: "0.8125rem",
                fontWeight: 600,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
              }}
            >
              🖨️ Print / Save PDF
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
