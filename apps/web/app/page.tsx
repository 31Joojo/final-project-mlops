"use client";

import { useState } from "react";

type PredictResponse = {
  prediction: number;
  probability: number | null;
  model_name: string;
  model_stage: string;
  model_version: string | null;
};

export default function Page() {
  const [salesPerson, setSalesPerson] = useState("Alice");
  const [country, setCountry] = useState("FR");
  const [product, setProduct] = useState("Dark Chocolate");
  const [boxesShipped, setBoxesShipped] = useState(10);
  const [date, setDate] = useState("2024-02-01");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  async function onPredict(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload = {
        sales_person: salesPerson,
        country,
        product,
        boxes_shipped: boxesShipped,
        date, // expected format: YYYY-MM-DD
      };

      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`API error ${resp.status}: ${text}`);
      }

      setResult((await resp.json()) as PredictResponse);
    } catch (err: any) {
      setError(err?.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 720, margin: "40px auto", padding: 16 }}>
      <h1>MLOps Final Project</h1>

      <form
        onSubmit={onPredict}
        style={{
          display: "grid",
          gap: 12,
          padding: 16,
          border: "1px solid #ddd",
          borderRadius: 12,
        }}
      >
        <label>
          Sales person
          <input
            value={salesPerson}
            onChange={(e) => setSalesPerson(e.target.value)}
            style={{ width: "100%", padding: 8, marginTop: 6 }}
          />
        </label>

        <label>
          Country
          <input
            value={country}
            onChange={(e) => setCountry(e.target.value)}
            style={{ width: "100%", padding: 8, marginTop: 6 }}
          />
        </label>

        <label>
          Product
          <input
            value={product}
            onChange={(e) => setProduct(e.target.value)}
            style={{ width: "100%", padding: 8, marginTop: 6 }}
          />
        </label>

        <label>
          Boxes shipped
          <input
            type="number"
            min={0}
            value={boxesShipped}
            onChange={(e) => setBoxesShipped(parseInt(e.target.value || "0", 10))}
            style={{ width: "100%", padding: 8, marginTop: 6 }}
          />
        </label>

        <label>
          Date
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            style={{ width: "100%", padding: 8, marginTop: 6 }}
          />
        </label>

        <button type="submit" disabled={loading} style={{ padding: 10, borderRadius: 10, cursor: "pointer" }}>
          {loading ? "Predicting…" : "Predict"}
        </button>
      </form>

      {error && (
        <pre style={{ marginTop: 16, color: "crimson", whiteSpace: "pre-wrap" }}>
          {error}
        </pre>
      )}
      {result && (
        <pre style={{ marginTop: 16, background: "#fafafa", padding: 12, borderRadius: 10 }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </main>
  );
}
