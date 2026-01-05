import React from "react";
import { predict, explain, health } from "./api";

import InputForm from "./components/InputForm";
import PredictionPanel from "./components/PredictionPanel";
import ExplainPanel from "./components/ExplainPanel";
import StatusPanel from "./components/StatusPanel";

export default function App() {
  const [prediction, setPrediction] = React.useState(null);
  const [shap, setShap] = React.useState(null);
  const [status, setStatus] = React.useState(null);
  const [loading, setLoading] = React.useState(false);

  const run = async (input) => {
    try {
      setLoading(true);

      const p = await predict(input);
      setPrediction(p.data);

      const e = await explain(input);
      setShap(e.data.shap_values);

      const s = await health();
      setStatus(s.data);
    } catch (err) {
      console.error("Analysis failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h2>Trustworthy NWDAF Dashboard</h2>

      <div className="grid">
        {/* Traffic Input */}
        <InputForm onSubmit={run} />

        {/* Prediction + Health */}
        <div>
          <PredictionPanel
            result={prediction}
            shap={shap}   
            loading={loading}
          />
          <StatusPanel status={status} />
        </div>

        {/* SHAP Explainability */}
        <div className="full-width">
          <ExplainPanel shap={shap} />
        </div>
      </div>
    </div>
  );
}
