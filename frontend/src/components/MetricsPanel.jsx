/* ═══════════════════════════════════════════
   MetricsPanel — Interactive Model Performance Dashboard
   Displays AUC-ROC, Accuracy, Sensitivity, Specificity,
   Confusion Matrix, and explanatory notes.
   ═══════════════════════════════════════════ */

/* ── Validated Model Metrics (from Kaggle evaluation) ── */
const METRICS = {
  accuracy: 0.7514,
  precision: 0.6763,
  recall: 0.7477,       // Sensitivity / TPR
  specificity: 0.7539,  // TNR
  f1: 0.7102,
  aucRoc: 0.8417,
  confusion: { tn: 239, fp: 78, fn: 55, tp: 163 },
  testSetSize: 535,
  malignantCount: 218,
  benignCount: 317,
}

function MetricCard({ label, value, description, color = "teal" }) {
  const colorMap = {
    teal: { text: "text-teal-400", bg: "bg-teal-500/8", border: "border-teal-500/15" },
    cyan: { text: "text-cyan-400", bg: "bg-cyan-500/8", border: "border-cyan-500/15" },
    emerald: { text: "text-emerald-400", bg: "bg-emerald-500/8", border: "border-emerald-500/15" },
    amber: { text: "text-amber-400", bg: "bg-amber-500/8", border: "border-amber-500/15" },
    rose: { text: "text-rose-400", bg: "bg-rose-500/8", border: "border-rose-500/15" },
  }
  const c = colorMap[color] || colorMap.teal

  return (
    <div className={`${c.bg} border ${c.border} rounded-xl p-3.5 text-center transition-transform duration-200 hover:scale-[1.02]`}>
      <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-1.5">{label}</p>
      <p className={`text-2xl font-bold font-mono ${c.text} tabular-nums`}>
        {typeof value === "number" ? (value * 100).toFixed(1) + "%" : value}
      </p>
      {description && (
        <p className="text-[10px] text-slate-600 mt-1">{description}</p>
      )}
    </div>
  )
}

function ConfusionMatrix() {
  const { confusion: cm } = METRICS

  return (
    <div>
      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
        Confusion Matrix
      </h4>
      <div className="glass-inset rounded-xl p-4">
        {/* Column headers */}
        <div className="grid grid-cols-[80px_1fr_1fr] gap-2 mb-2">
          <div />
          <p className="text-center text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Pred. Benign</p>
          <p className="text-center text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Pred. Malignant</p>
        </div>

        {/* Row 1: Actual Benign */}
        <div className="grid grid-cols-[80px_1fr_1fr] gap-2 mb-2">
          <div className="flex items-center">
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider leading-tight">Actual Benign</p>
          </div>
          <div className="matrix-cell matrix-cell-tn tooltip-container">
            {cm.tn}
            <span className="tooltip-text">True Negative — Correctly identified as Benign</span>
          </div>
          <div className="matrix-cell matrix-cell-fp tooltip-container">
            {cm.fp}
            <span className="tooltip-text">False Positive — Benign misclassified as Malignant</span>
          </div>
        </div>

        {/* Row 2: Actual Malignant */}
        <div className="grid grid-cols-[80px_1fr_1fr] gap-2">
          <div className="flex items-center">
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider leading-tight">Actual Malignant</p>
          </div>
          <div className="matrix-cell matrix-cell-fn tooltip-container">
            {cm.fn}
            <span className="tooltip-text">False Negative — Malignant misclassified as Benign</span>
          </div>
          <div className="matrix-cell matrix-cell-tp tooltip-container">
            {cm.tp}
            <span className="tooltip-text">True Positive — Correctly identified as Malignant</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function MetricsPanel() {
  return (
    <div className="space-y-5 tab-content">
      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 stagger-children">
        <MetricCard label="AUC-ROC" value={METRICS.aucRoc} description="Area Under Curve" color="teal" />
        <MetricCard label="Accuracy" value={METRICS.accuracy} description="Overall correct" color="cyan" />
        <MetricCard label="Sensitivity" value={METRICS.recall} description="True Positive Rate" color="emerald" />
        <MetricCard label="Specificity" value={METRICS.specificity} description="True Negative Rate" color="amber" />
        <MetricCard label="Precision" value={METRICS.precision} description="Positive Predictive" color="teal" />
        <MetricCard label="F1-Score" value={METRICS.f1} description="Harmonic Mean" color="cyan" />
      </div>

      {/* Confusion Matrix */}
      <ConfusionMatrix />

      {/* Evaluation Context */}
      <div className="glass-inset rounded-xl p-4 space-y-3">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
          Evaluation Context
        </h4>
        <div className="grid grid-cols-3 gap-3">
          <div className="text-center">
            <p className="text-lg font-bold font-mono text-slate-300">{METRICS.testSetSize}</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Test Images</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold font-mono text-emerald-400">{METRICS.benignCount}</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Benign</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold font-mono text-rose-400">{METRICS.malignantCount}</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Malignant</p>
          </div>
        </div>

        <div className="border-t border-slate-700/30 pt-3">
          <p className="text-[11px] text-slate-500 leading-relaxed">
            <span className="text-slate-400 font-medium">Why accuracy alone is insufficient:</span>{" "}
            In screening tasks with class imbalance, a model could achieve high accuracy by simply predicting the majority class.
            Sensitivity (Recall) ensures malignant cases are not missed, while Specificity minimizes false alarms.
            The AUC-ROC of <span className="text-teal-400 font-mono font-semibold">0.84</span> indicates strong discriminative ability across all thresholds.
          </p>
        </div>
      </div>
    </div>
  )
}
