"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  Sparkles,
  Cpu,
  Binary,
  RotateCw,
  AlertCircle,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ArrowLeft,
  Clock,
  ChevronRight,
  Info,
  Activity,
} from "lucide-react";
import { predictSentiment, checkBackendHealth, API_BASE_URL } from "../../services/api";

function renderModelCard(res, title, IconComponent, accentColor = "indigo") {
  if (!res) return null;

  if (res.unavailable) {
    return (
      <div className="rounded-3xl border border-amber-300 bg-amber-50/80 p-6 shadow-sm dark:border-amber-900/60 dark:bg-amber-950/30">
        <div className="flex items-center justify-between border-b border-amber-200/70 pb-3 dark:border-amber-900/50">
          <div className="flex items-center gap-2">
            <IconComponent className="h-4 w-4 text-amber-600 dark:text-amber-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-amber-900 dark:text-amber-200">
              {title}
            </span>
          </div>
          <span className="rounded-full bg-amber-200/60 px-2 py-0.5 text-[10px] font-bold text-amber-900 dark:bg-amber-900/60 dark:text-amber-200">
            UNAVAILABLE
          </span>
        </div>
        <div className="mt-4 flex items-start gap-2.5 text-xs text-amber-900 dark:text-amber-200">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-amber-600" />
          <p className="leading-relaxed">
            {res.error || "This model is not loaded on the backend server."}
          </p>
        </div>
      </div>
    );
  }

  const isPositive = res.sentiment === "POSITIVE";
  const calProbNeg = res.calibratedProbabilities?.NEGATIVE ?? res.probabilities?.NEGATIVE ?? 0;
  const calProbPos = res.calibratedProbabilities?.POSITIVE ?? res.probabilities?.POSITIVE ?? 0;
  const rawProbPct =
    res.modelProbability != null ? (res.modelProbability * 100).toFixed(2) : null;
  const borderClass =
    accentColor === "indigo"
      ? "border-indigo-200/90 dark:border-indigo-900/60"
      : "border-slate-200 dark:border-slate-800";

  return (
    <div className={`rounded-3xl border ${borderClass} bg-white p-6 shadow-sm dark:bg-slate-900`}>
      <div className="flex items-center justify-between border-b border-slate-100 pb-3 dark:border-slate-800">
        <div className="flex items-center gap-2">
          <IconComponent
            className={`h-4 w-4 ${
              accentColor === "indigo"
                ? "text-indigo-600 dark:text-indigo-400"
                : "text-slate-600 dark:text-slate-400"
            }`}
          />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-200">
            {title}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {res.confidenceBand && (
            <span
              className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                res.confidenceBand === "HIGH"
                  ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300"
                  : res.confidenceBand === "MODERATE"
                  ? "bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300"
                  : "bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300"
              }`}
            >
              {res.confidenceBand} CONF
            </span>
          )}
          <span className="flex items-center gap-1 font-mono text-[11px] text-slate-400">
            <Clock className="h-3 w-3" />
            <span>{res.latencyMs}ms</span>
          </span>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between">
        <div>
          <span className="text-[11px] uppercase font-bold text-slate-400">
            Predicted Class {res.rawClassId !== undefined && `(ID: ${res.rawClassId})`}
          </span>
          <div className="mt-1 flex items-center gap-2">
            {isPositive ? (
              <CheckCircle className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            ) : (
              <XCircle className="h-5 w-5 text-rose-600 dark:text-rose-400" />
            )}
            <span
              className={`text-2xl font-black ${
                isPositive
                  ? "text-emerald-700 dark:text-emerald-400"
                  : "text-rose-700 dark:text-rose-400"
              }`}
            >
              {res.sentiment}
            </span>
          </div>
        </div>

        <div className="text-right">
          <span className="text-[11px] uppercase font-bold text-slate-400">
            Calibrated Confidence
          </span>
          <div className="mt-1 font-mono text-2xl font-black text-slate-900 dark:text-white">
            {res.confidencePercentage}%
          </div>
          {rawProbPct && (
            <div className="font-mono text-[11px] text-slate-400">
              Raw Prob: {rawProbPct}%
            </div>
          )}
        </div>
      </div>

      {/* Confidence Meter */}
      <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            isPositive ? "bg-emerald-500" : "bg-rose-500"
          }`}
          style={{ width: `${res.confidencePercentage}%` }}
        />
      </div>

      {/* Class Probabilities */}
      <div className="mt-4 grid grid-cols-2 gap-3 rounded-xl bg-slate-50/80 p-3 text-xs dark:bg-slate-950/60">
        <div>
          <span className="text-[10px] font-bold uppercase text-rose-600 dark:text-rose-400">
            Negative (0)
          </span>
          <div className="font-mono font-bold text-slate-800 dark:text-slate-200">
            {(calProbNeg * 100).toFixed(2)}%
          </div>
        </div>
        <div className="text-right">
          <span className="text-[10px] font-bold uppercase text-emerald-600 dark:text-emerald-400">
            Positive (1)
          </span>
          <div className="font-mono font-bold text-slate-800 dark:text-slate-200">
            {(calProbPos * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      {res.hasOutOfVocabularyWarning && (
        <div className="mt-3 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-300">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
          <span>No matching TF-IDF vocabulary n-grams found; prediction reflects class baseline.</span>
        </div>
      )}

      {/* Tokens / Features */}
      {res.tokens && res.tokens.length > 0 && (
        <div className="mt-4 border-t border-slate-100 pt-3 dark:border-slate-800">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-2">
            {res.modelKey === "bert" ? "WordPiece Sub-tokens:" : "Matched TF-IDF Features:"}
          </span>
          <div className="flex flex-wrap gap-1.5">
            {res.tokens.map((t, idx) => (
              <span
                key={idx}
                className="rounded-md border border-indigo-200 bg-indigo-50/70 px-2 py-0.5 font-mono text-[11px] font-medium text-indigo-700 dark:border-indigo-800 dark:bg-indigo-950/60 dark:text-indigo-300"
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function AnalyzePage() {
  const [text, setText] = useState(
    "I am really happy with the service. Your support team solved my problem quickly."
  );
  const [model, setModel] = useState("both");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkBackendHealth().then(setHealth);
  }, []);

  const sampleTickets = [
    {
      category: "Customer Delight",
      label: "Positive Praise",
      text: "I am really happy with the service. Your support team solved my problem quickly.",
    },
    {
      category: "Escalation",
      label: "Service Outage",
      text: "The payment gateway failed during checkout and my card was charged twice. Terrible experience!",
    },
    {
      category: "Mixed Critique",
      label: "Mixed Review",
      text: "The movie had excellent acting but a terrible story.",
    },
    {
      category: "Neutral Inquiry",
      label: "Factual Notice",
      text: "The account was created successfully.",
    },
  ];

  const handleTextChange = (val) => {
    setText(val);
    if (error) setError(null);
  };

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictSentiment(text, model);
      if (model === "both") {
        setResult(res);
      } else {
        setResult({
          [model]: res,
        });
      }
    } catch (err) {
      setResult(null);
      setError(err?.message || "Failed to classify text.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="py-10 md:py-16">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Breadcrumb / Back Link */}
        <div className="mb-6 flex items-center gap-2 text-xs font-medium text-slate-500">
          <Link href="/" className="hover:text-indigo-600 flex items-center gap-1">
            <ArrowLeft className="h-3.5 w-3.5" />
            <span>Back to Landing</span>
          </Link>
          <ChevronRight className="h-3.5 w-3.5" />
          <span className="text-slate-900 dark:text-white font-semibold">
            Sentiment Analyzer Studio
          </span>
        </div>

        {/* Page Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-200/80 pb-6 dark:border-slate-800">
          <div>
            <div className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300">
              <Sparkles className="h-3.5 w-3.5 text-indigo-600" />
              <span>Interactive Decision-Support Studio</span>
            </div>
            <h1 className="mt-2 text-2xl font-extrabold text-slate-900 sm:text-3xl dark:text-white">
              Sentiment Analyzer
            </h1>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
              Analyze text using Fine-Tuned BERT and compare side-by-side with TF-IDF + Logistic Regression.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {health && (
              <div
                className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-semibold ${
                  health.status === "ok"
                    ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300"
                    : health.status === "degraded"
                    ? "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900/50 dark:bg-amber-950/40 dark:text-amber-300"
                    : "border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-300"
                }`}
                title={`API: ${API_BASE_URL}`}
              >
                <Activity className="h-3.5 w-3.5" />
                <span>
                  {health.status === "ok"
                    ? "API Online (BERT + TF-IDF Ready)"
                    : health.status === "degraded"
                    ? `Degraded (${health.available_models?.join(", ") || "partial"})`
                    : "Backend Offline"}
                </span>
              </div>
            )}
            <Link
              href="/model"
              className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-2xs hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
            >
              <Info className="h-3.5 w-3.5" />
              <span>Model Card</span>
            </Link>
            <Link
              href="/performance"
              className="inline-flex items-center gap-1.5 rounded-xl bg-indigo-600 px-3.5 py-2 text-xs font-semibold text-white shadow-2xs hover:bg-indigo-500"
            >
              <span>Benchmarks</span>
            </Link>
          </div>
        </div>

        {/* Studio Layout */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Input and Configuration (7 cols) */}
          <div className="lg:col-span-7 space-y-6">
            {/* Input Card */}
            <div className="rounded-3xl border border-slate-200/90 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <div className="flex items-center justify-between">
                <label
                  htmlFor="studio-input"
                  className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400"
                >
                  Customer Message / Support Ticket
                </label>
                <span className="font-mono text-xs text-slate-400">
                  {text.length}/500 chars
                </span>
              </div>

              <textarea
                id="studio-input"
                rows={5}
                value={text}
                onChange={(e) => handleTextChange(e.target.value)}
                placeholder="Paste customer email, review snippet, or support message..."
                maxLength={500}
                className="mt-3 w-full rounded-2xl border border-slate-200 bg-slate-50/50 p-4 text-sm leading-relaxed text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 dark:border-slate-800 dark:bg-slate-950/60 dark:text-white dark:focus:bg-slate-950"
              />

              {/* Sample Ticket Shortcuts */}
              <div className="mt-4">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-2">
                  Pre-configured Scenarios:
                </span>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {sampleTickets.map((sample) => (
                    <button
                      key={sample.label}
                      type="button"
                      onClick={() => {
                        handleTextChange(sample.text);
                        setResult(null);
                      }}
                      className="rounded-xl border border-slate-200 bg-slate-50 p-2.5 text-left transition-all hover:border-indigo-300 hover:bg-white dark:border-slate-800 dark:bg-slate-950 dark:hover:border-indigo-700"
                    >
                      <span className="block text-[10px] font-bold text-indigo-600 dark:text-indigo-400 uppercase">
                        {sample.category}
                      </span>
                      <span className="mt-0.5 block text-xs font-semibold text-slate-800 dark:text-slate-200">
                        {sample.label}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Model Choice & Run Action */}
              <div className="mt-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-t border-slate-100 pt-5 dark:border-slate-800">
                <div className="flex items-center gap-1 rounded-xl bg-slate-100 p-1 dark:bg-slate-800">
                  <button
                    type="button"
                    onClick={() => setModel("bert")}
                    className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                      model === "bert"
                        ? "bg-white text-indigo-600 shadow-2xs dark:bg-slate-900 dark:text-white"
                        : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
                    }`}
                  >
                    BERT Only
                  </button>
                  <button
                    type="button"
                    onClick={() => setModel("tfidf")}
                    className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                      model === "tfidf"
                        ? "bg-white text-indigo-600 shadow-2xs dark:bg-slate-900 dark:text-white"
                        : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
                    }`}
                  >
                    TF-IDF Only
                  </button>
                  <button
                    type="button"
                    onClick={() => setModel("both")}
                    className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                      model === "both"
                        ? "bg-white text-indigo-600 shadow-2xs dark:bg-slate-900 dark:text-white"
                        : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
                    }`}
                  >
                    Compare Both
                  </button>
                </div>

                <button
                  type="button"
                  disabled={loading || !text.trim()}
                  onClick={handleAnalyze}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-6 py-2.5 text-sm font-semibold text-white shadow-sm shadow-indigo-600/30 transition-all hover:shadow-indigo-600/40 disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <RotateCw className="h-4 w-4 animate-spin" />
                      <span>Computing Logits...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      <span>Run Classification</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Educational Disclaimer Banner */}
            <div className="rounded-2xl border border-amber-200/80 bg-amber-50/70 p-4 dark:border-amber-900/50 dark:bg-amber-950/30">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400 mt-0.5" />
                <div className="text-xs text-amber-900 dark:text-amber-200 leading-relaxed">
                  <strong>Evaluation Scope:</strong> Both models are trained on the binary SST-2 movie review corpus
                  (0 = NEGATIVE, 1 = POSITIVE). Neutral or mixed texts will still resolve to one of two binary classes.
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Live Inference Output (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            {error && (
              <div
                role="alert"
                className="rounded-3xl border border-rose-300 bg-rose-50/90 p-6 shadow-sm dark:border-rose-900/60 dark:bg-rose-950/30"
              >
                <div className="flex items-start gap-3">
                  <XCircle className="h-5 w-5 shrink-0 text-rose-600 dark:text-rose-400 mt-0.5" />
                  <div>
                    <h3 className="text-sm font-bold text-rose-900 dark:text-rose-200">
                      Classification Request Failed
                    </h3>
                    <p className="mt-1 text-xs leading-relaxed text-rose-800 dark:text-rose-300">
                      {error}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {!result && !error && (
              <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50/60 p-8 text-center dark:border-slate-800 dark:bg-slate-900/40">
                <Sparkles className="mx-auto h-8 w-8 text-indigo-500/70" />
                <h3 className="mt-3 text-sm font-bold text-slate-800 dark:text-slate-200">
                  Ready for Live Inference
                </h3>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Enter your text or pick a scenario and click <strong>Run Classification</strong> to inspect live model predictions, raw probabilities, calibrated confidence, and tokens.
                </p>
              </div>
            )}

            {result && (model === "bert" || model === "both") &&
              renderModelCard(result.bert, "BERT Transformer Output", Cpu, "indigo")}

            {result && (model === "tfidf" || model === "both") &&
              renderModelCard(result.tfidf, "TF-IDF + Logistic Regression Output", Binary, "slate")}

            {result &&
              model === "both" &&
              result.bert &&
              result.tfidf &&
              !result.bert.unavailable &&
              !result.tfidf.unavailable &&
              result.bert.sentiment !== result.tfidf.sentiment && (
                <div className="rounded-2xl border border-indigo-200/80 bg-indigo-50/70 p-4 text-xs text-indigo-950 dark:border-indigo-900/60 dark:bg-indigo-950/30 dark:text-indigo-200 leading-relaxed">
                  <strong>Why Models Disagreed:</strong> BERT evaluates full sentence context and negations, whereas TF-IDF sums isolated SST-2 movie-review n-gram weights and can misread negated phrases or out-of-domain support terms.
                </div>
              )}
          </div>
        </div>
      </div>
    </div>
  );
}
