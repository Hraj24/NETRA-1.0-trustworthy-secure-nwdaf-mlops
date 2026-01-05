import React from "react";

export default function InputForm({ onSubmit }) {
  const [form, setForm] = React.useState({
    time_of_day: 12,
    slice_type: "URLLC",
    jitter: 5,
    packet_loss: 0.2,
    throughput: 200
  });

  return (
    <div className="card">
      <h3>Traffic Input</h3>

      <div className="form-grid">
        {Object.keys(form).map((k) => (
          <div key={k} className="form-group">
            <label>{k.replace("_", " ")}</label>
            <input
              value={form[k]}
              onChange={(e) =>
                setForm({ ...form, [k]: e.target.value })
              }
            />
          </div>
        ))}
      </div>

      <button onClick={() => onSubmit(form)}>
        Run Analysis
      </button>
    </div>
  );
}
