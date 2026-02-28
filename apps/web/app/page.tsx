"use client";

import { useEffect, useState } from "react";
import { Fredoka } from "next/font/google";

const titleFont = Fredoka({
  subsets: ["latin"],
  weight: ["600", "700"],
});

type PredictResponse = {
  prediction: number;
  probability: number | null;
  model_name: string;
  model_stage: string;
  model_version: string | null;
};

type OptionsResponse = {
  sales_persons: string[];
  countries: string[];
  products: string[];
};

const TEAM_NAMES = [
  "BAUDET Quentin",
  "CARDONA Quentin",
  "LARMAILLARD-NOIREN Joris",
];

export default function Page() {
  const [salesPerson, setSalesPerson] = useState("Alice");
  const [country, setCountry] = useState("FR");
  const [product, setProduct] = useState("Dark Chocolate");
  const [boxesShipped, setBoxesShipped] = useState(10);
  const [date, setDate] = useState("2026-02-28");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [optionsError, setOptionsError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadOptions() {
      setOptionsLoading(true);
      setOptionsError(null);

      try {
        const resp = await fetch("/api/options");
        if (!resp.ok) {
          const txt = await resp.text();
          throw new Error(`Options error ${resp.status}: ${txt}`);
        }

        const data = (await resp.json()) as OptionsResponse;
        if (cancelled) return;

        setOptions(data);

        // If current values not present default to first available option
        if (
          data.sales_persons?.length &&
          !data.sales_persons.includes(salesPerson)
        ) {
          setSalesPerson(data.sales_persons[0]);
        }
        if (data.countries?.length && !data.countries.includes(country)) {
          setCountry(data.countries[0]);
        }
        if (data.products?.length && !data.products.includes(product)) {
          setProduct(data.products[0]);
        }
      } catch (e: any) {
        if (cancelled) return;
        setOptionsError(e?.message ?? "Failed to load options");
        setOptions(null);
      } finally {
        if (!cancelled) setOptionsLoading(false);
      }
    }

    loadOptions();
    return () => {
      cancelled = true;
    };
  }, []);

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
        date,
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

  const probaPct =
    result?.probability == null
      ? null
      : Math.round(result.probability * 1000) / 10;

  const predictedLabel =
    result == null
      ? null
      : result.prediction === 1
        ? "high amount"
        : "low amount";

  const subtitle =
    "Estimate the probability that a sale will be classified as high value based on the context.";

  const disableDropdowns = optionsLoading || options == null;

  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        backgroundImage:
          "url('https://images.unsplash.com/photo-1646151067116-80a86e5eee69?q=80&w=2103&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundAttachment: "fixed",
      }}
    >
      {/* Content wrapper */}
      <div style={{ flex: 1, padding: "32px 16px 28px" }}>
        <div style={{ maxWidth: 980, margin: "0 auto" }}>
          {/* Header */}
          <header style={{ marginBottom: 18 }}>
            <div
              className={titleFont.className}
              style={{
                fontFamily: `${titleFont.style.fontFamily}, "Arial Rounded MT Bold", "Arial Rounded MT", Arial, sans-serif`,
                fontSize: 44,
                lineHeight: 1.05,
                letterSpacing: -0.5,
                color: "#111",
                textAlign: "center",
                WebkitTextStroke: "1px rgba(255,255,255,0.65)",
                textShadow: "0 6px 18px rgba(0,0,0,0.12)",
              }}
            >
              Chocolate Sales Prediction
            </div>

            <div
              style={{
                marginTop: 10,
                textAlign: "center",
                color: "rgba(0,0,0,0.78)",
                fontSize: 16,
                fontWeight: "bold",
                maxWidth: 760,
                marginLeft: "auto",
                marginRight: "auto",
                textShadow: "0 2px 10px rgba(255,255,255,0.35)",
              }}
            >
              {subtitle}
            </div>
          </header>

          {/* Semi-transparent card */}
          <section
            style={{
              background: "rgba(255, 255, 255, 0.55)",
              backdropFilter: "blur(10px)",
              border: "1px solid rgba(255,255,255,0.35)",
              boxShadow: "0 20px 60px rgba(0,0,0,0.20)",
              borderRadius: 16,
              padding: 18,
            }}
          >
            <form onSubmit={onPredict}>
              {/* Inner wrapper so grid never exceeds the card */}
              <div style={{ width: "100%", maxWidth: 920, margin: "0 auto" }}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                    columnGap: 28,
                    rowGap: 16,
                    alignItems: "start",
                  }}
                >
                  {/* Left column - 3 items */}
                  <div style={{ display: "grid", gap: 12 }}>
                    <label style={{ display: "grid", gap: 6 }}>
                      <span style={{ fontWeight: 700 }}>Sales person</span>
                      <select
                        value={salesPerson}
                        onChange={(e) => setSalesPerson(e.target.value)}
                        style={inputStyle}
                        disabled={disableDropdowns}
                      >
                        {optionsLoading && <option>Loading…</option>}
                        {!optionsLoading &&
                          options?.sales_persons?.length === 0 && (
                            <option>No sales persons</option>
                          )}
                        {!optionsLoading &&
                          options?.sales_persons?.map((v) => (
                            <option key={v} value={v}>
                              {v}
                            </option>
                          ))}
                      </select>
                    </label>

                    <label style={{ display: "grid", gap: 6 }}>
                      <span style={{ fontWeight: 700 }}>Country</span>
                      <select
                        value={country}
                        onChange={(e) => setCountry(e.target.value)}
                        style={inputStyle}
                        disabled={disableDropdowns}
                      >
                        {optionsLoading && <option>Loading...</option>}
                        {!optionsLoading &&
                          options?.countries?.length === 0 && (
                            <option>No countries</option>
                          )}
                        {!optionsLoading &&
                          options?.countries?.map((v) => (
                            <option key={v} value={v}>
                              {v}
                            </option>
                          ))}
                      </select>
                    </label>

                    <label style={{ display: "grid", gap: 6 }}>
                      <span style={{ fontWeight: 700 }}>Product</span>
                      <select
                        value={product}
                        onChange={(e) => setProduct(e.target.value)}
                        style={inputStyle}
                        disabled={disableDropdowns}
                      >
                        {optionsLoading && <option>Loading…</option>}
                        {!optionsLoading && options?.products?.length === 0 && (
                          <option>No products</option>
                        )}
                        {!optionsLoading &&
                          options?.products?.map((v) => (
                            <option key={v} value={v}>
                              {v}
                            </option>
                          ))}
                      </select>
                    </label>
                  </div>

                  {/* Right column - 2 items */}
                  <div style={{ display: "grid", gap: 12 }}>
                    <label style={{ display: "grid", gap: 6 }}>
                      <span style={{ fontWeight: 700 }}>Boxes shipped</span>
                      <input
                        type="number"
                        min={0}
                        value={boxesShipped}
                        onChange={(e) =>
                          setBoxesShipped(parseInt(e.target.value || "0", 10))
                        }
                        style={inputStyle}
                      />
                    </label>

                    <label style={{ display: "grid", gap: 6 }}>
                      <span style={{ fontWeight: 700 }}>Date</span>
                      <input
                        type="date"
                        value={date}
                        onChange={(e) => setDate(e.target.value)}
                        style={inputStyle}
                      />
                    </label>
                  </div>

                  {/* Centered button spans both columns */}
                  <div
                    style={{
                      gridColumn: "1 / -1",
                      display: "flex",
                      justifyContent: "center",
                      marginTop: 6,
                    }}
                  >
                    <button
                      type="submit"
                      disabled={loading}
                      style={buttonStyle(loading)}
                    >
                      {loading ? "Predicting…" : "Predict"}
                    </button>
                  </div>
                </div>
              </div>
            </form>

            {/* Error */}
            {error && (
              <div
                style={{
                  marginTop: 14,
                  padding: 12,
                  borderRadius: 12,
                  background: "rgba(220, 20, 60, 0.12)",
                }}
              >
                <div
                  style={{ fontWeight: 800, color: "crimson", marginBottom: 6 }}
                >
                  Something went wrong
                </div>
                <pre
                  style={{
                    margin: 0,
                    color: "crimson",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {error}
                </pre>
              </div>
            )}

            {/* Results */}
            {result && (
              <div style={{ width: "100%", maxWidth: 920, margin: "0 auto" }}>
                <div
                  style={{
                    marginTop: 16,
                    padding: 14,
                    borderRadius: 14,
                    background: "rgba(255,255,255,0.65)",
                    border: "1px solid rgba(0,0,0,0.06)",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 12,
                    }}
                  >
                    <div style={{ fontWeight: 900, fontSize: 16 }}>Result</div>
                    <div
                      style={{
                        display: "flex",
                        gap: 8,
                        flexWrap: "wrap",
                        justifyContent: "flex-end",
                      }}
                    >
                      <Badge label={result.model_name} />
                      <Badge label={result.model_stage} />
                      {result.model_version ? (
                        <Badge label={`v${result.model_version}`} />
                      ) : (
                        <Badge label={"v?"} />
                      )}
                    </div>
                  </div>

                  <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
                    <div style={{ fontSize: 14 }}>
                      <strong>Predicted category :</strong> {predictedLabel}
                      {probaPct != null ? (
                        <>
                          {" "}
                          — probability <strong>"high amount"</strong> :{" "}
                          <strong>{probaPct}%</strong>
                        </>
                      ) : null}
                    </div>

                    {/* Progress bar */}
                    {probaPct != null && (
                      <div>
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            fontSize: 12,
                            color: "rgba(0,0,0,0.65)",
                          }}
                        >
                          <span>Low amount</span>
                          <span>High amount</span>
                        </div>
                        <div
                          style={{
                            height: 10,
                            borderRadius: 999,
                            background: "rgba(0,0,0,0.10)",
                            overflow: "hidden",
                            marginTop: 6,
                          }}
                        >
                          <div
                            style={{
                              width: `${Math.min(100, Math.max(0, probaPct))}%`,
                              height: "100%",
                              borderRadius: 999,
                              background: "rgba(0,0,0,0.55)",
                            }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Interpretation text matching your business meaning */}
                    {probaPct != null && (
                      <div
                        style={{
                          fontSize: 13,
                          color: "rgba(0,0,0,0.78)",
                          lineHeight: 1.45,
                        }}
                      >
                        For this combination (<strong>{salesPerson}</strong> /{" "}
                        <strong>{country}</strong> / <strong>{product}</strong>{" "}
                        / <strong>{date}</strong> /{" "}
                        <strong>{boxesShipped}</strong> boxes shipped), the
                        model estimates at <strong>{probaPct}%</strong> the
                        probability that the sale is in the{" "}
                        <strong>"high amount"</strong> (above the threshold
                        defined during training).
                      </div>
                    )}

                    <div style={{ fontSize: 12, color: "rgba(0,0,0,0.60)" }}>
                      Note: “high amount” corresponds to the positive class
                      defined during training (above the threshold).
                    </div>
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      </div>

      {/* Footer */}
      <footer
        style={{
          background: "#0b0b0b",
          color: "white",
          padding: "14px 16px",
        }}
      >
        <div style={{ maxWidth: 980, margin: "0 auto", textAlign: "center" }}>
          {TEAM_NAMES.join(" • ")}
        </div>
      </footer>
    </main>
  );
}

function Badge({ label }: { label: string }) {
  return (
    <span
      style={{
        fontSize: 12,
        padding: "6px 10px",
        borderRadius: 999,
        background: "rgba(0,0,0,0.10)",
        border: "1px solid rgba(0,0,0,0.10)",
      }}
    >
      {label}
    </span>
  );
}

const inputStyle: React.CSSProperties = {
  width: "100%",
  minWidth: 0,
  padding: "10px 12px",
  borderRadius: 12,
  border: "1px solid rgba(0,0,0,0.12)",
  outline: "none",
  background: "rgba(255,255,255,0.88)",
};

function buttonStyle(loading: boolean): React.CSSProperties {
  return {
    padding: "10px 18px",
    borderRadius: 999,
    cursor: loading ? "not-allowed" : "pointer",
    border: "1px solid rgba(0,0,0,0.20)",
    background: loading ? "rgba(0,0,0,0.35)" : "rgba(0,0,0,0.70)",
    color: "white",
    fontWeight: 800,
    minWidth: 180,
    boxShadow: "0 10px 25px rgba(0,0,0,0.18)",
  };
}
