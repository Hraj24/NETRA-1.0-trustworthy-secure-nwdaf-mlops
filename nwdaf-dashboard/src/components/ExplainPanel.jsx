import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function ExplainPanel({ shap }) {
  if (!shap) return null;

  const data = Object.entries(shap).map(([k, v]) => ({
    feature: k,
    value: v
  }));

  return (
    <div className="card">
      <h3>SHAP Explainability</h3>

      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data}>
          <XAxis dataKey="feature" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill="#8b5cf6" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
