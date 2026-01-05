import React from "react";

/* =========================
   SHAP Helper Functions
   ========================= */

function getShapVariance(shap) {
  if (!shap) return null;

  const values = Object.values(shap);
  if (values.length === 0) return null;

  const mean =
    values.reduce((sum, v) => sum + v, 0) / values.length;

  const variance =
    values.reduce(
      (sum, v) => sum + Math.pow(v - mean, 2),
      0
    ) / values.length;

  return variance;
}

function getNormalizedVariance(variance, predictedLoad) {
  return variance / (predictedLoad ** 2);
}

function getConfidenceFromVariance(variance, predictedLoad) {
  if (!variance || !predictedLoad) return "Low";

  const normalizedVariance =
    variance / Math.pow(predictedLoad, 2);

  if (normalizedVariance < 0.05) return "High";
  if (normalizedVariance < 0.12) return "Medium";
  return "Low";
}




// function getConfidenceFromVariance(variance) {
//   if (variance < 500) return "High";
//   if (variance < 2000) return "Medium";
//   return "Low";
// }

/* =========================
   Structured Unified Explanation
   ========================= */

function generateUnifiedExplanation({ confidence, shap, input }) {
  if (!confidence || !shap) return null;

  // Top SHAP feature
  const topFeature = Object.entries(shap)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0][0];

  /* -------- Model Reasoning -------- */
  let modelReason = "";
  if (confidence === "High") {
    modelReason =
      "Feature contributions are stable and consistent across the prediction.";
  } else if (confidence === "Medium") {
    modelReason =
      `Moderate variation detected in feature contributions, particularly '${topFeature}'.`;
  } else {
    modelReason =
      `High sensitivity detected in feature contributions, especially '${topFeature}'.`;
  }

  /* -------- Rule-Based Network Reasons -------- */
  const ruleReasons = [];
  if (input) {
    if (Number(input.jitter) > 30)
      ruleReasons.push("High jitter observed");
    if (Number(input.packet_loss) > 2)
      ruleReasons.push("Packet loss detected");
    if (Number(input.throughput) > 800)
      ruleReasons.push("Heavy throughput usage");
  }

  if (ruleReasons.length === 0) {
    ruleReasons.push(
      "Network conditions appear stable with no dominant risk factors."
    );
  }

  return {
    confidenceSummary: `The model predicts the future network load with ${confidence.toLowerCase()} confidence.`,
    modelReason,
    ruleReasons
  };
}

/* =========================
   Component
   ========================= */

export default function PredictionPanel({
  result,
  shap,
  input,
  loading
}) {
  if (!result) return null;

  const variance = getShapVariance(shap);
  // const confidence = variance
  //   ? getConfidenceFromVariance(variance)
  //   : "Calculating";
  const confidence = getConfidenceFromVariance(
  variance,
  result.predicted_future_load
);

  const explanation = generateUnifiedExplanation({
    confidence,
    shap,
    input
  });

  return (
    <div className="card">
      <h3>Prediction</h3>

      <div className="metric">
        {loading
          ? "Running..."
          : result.predicted_future_load.toFixed(2)}
      </div>
      <div className="muted">Predicted Future Load</div>

      {/* Confidence Badge */}
      <div className={`confidence ${confidence.toLowerCase()}`}>
        Confidence: {confidence}
      </div>

      {variance && (
        <div className="muted">
          SHAP Variance: {variance.toFixed(2)}
        </div>
      )}

      {/* ---------- Structured Explanation ---------- */}
      {explanation && (
        <div className="explanation-box">
          <p className="explanation-title">
            {explanation.confidenceSummary}
          </p>

          <div className="explanation-section">
            <h4>🧠 Model Reasoning</h4>
            <p>{explanation.modelReason}</p>
          </div>

          <div className="explanation-section">
            <h4>⚙️ Network Conditions</h4>
            <ul>
              {explanation.ruleReasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {result.warning && (
        <div className="warning">{result.warning}</div>
      )}
    </div>
  );
}
