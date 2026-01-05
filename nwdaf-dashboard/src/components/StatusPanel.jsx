export default function StatusPanel({ status }) {
  if (!status) return null;

  const driftDetected = status.rollback_active;

  return (
    <div className="card">
      <h3>System Health</h3>

      <p>Model: {status.model_version}</p>
      <p>Detector: {status.primary_detector}</p>

      {/* Drift Badge */}
      <div className={`badge ${driftDetected ? "alert" : "ok"}`}>
        {driftDetected ? "Drift Detected" : "Stable"}
      </div>

      <p className="muted">
        Rollback Active: {driftDetected ? "YES" : "NO"}
      </p>
    </div>
  );
}
