"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Sparkles,
  Cpu,
  Binary,
  RotateCw,
  AlertCircle,
  CheckCircle,
  XCircle,
  ArrowLeft,
  Sliders,
  Layers,
  Clock,
  ChevronRight,
  Info,
} from "lucide-react";
import { predictSentiment } from "../../services/api";

export default function AnalyzePage() {
  const [text, setText] = useState(
    "I am really happy with the service. Your support team solved my problem quickly."
  );
  const [model, setModel] = useState("both"); // 'bert' | 'tfidf' | 'both'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState({
    bert: {
      sentiment: "POSITIVE",
      confidence: 0.95,
      confidencePercentage: 95,
      model: "BERT (bert-sst2-v1)",
      latencyMs: 38,
      tokens: ["I", "am", "really", "happy", "with", "the", "service", "."],
    },
    tfidf: {
      sentiment: "POSITIVE",
      confidence: 0.89,
      confidencePercentage: 89,
      model: "TF-IDF + Linear SVM",
      latencyMs: 6,
      tokens: ["happy", "service", "support", "team", "solved", "problem", "quickly"],
    },
  });

  const sampleTickets = [
    {
      category: "Customer Delight",
      label: "Resolved Quickly",
      text: "I am really happy with the service. Your support team solved my problem quickly.",
    },
    {
      category: "Escalation",
      label: "Service Outage",
      text: "The payment gateway failed during checkout and my card was charged twice. Terrible experience!",
    },
    {
      category: "Subtle Nuance",
      label: "Mixed Review",
      text: "The software has great potential, but the recent bug makes it difficult to finish urgent tasks.",
    },
    {
      category: "Neutral / Inquiry",
      label: "Account Access",
      text: "Could someone please guide me on how to reset my password on the user dashboard?",
    },
  ];

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      if (model === "both") {
        const res = await predictSentiment(text, "both");
        setResult(res);
      } else {
        const single = await predictSentiment(text, model);
        setResult({
          [model]: single,
        });
      }
    } catch (err) {
      console.error(err);
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
              Analyze text using Fine-Tuned BERT and compare side-by-side with TF-IDF + Linear SVM.
            </p>
          </div>

          <div className="flex items-center gap-3">
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
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste customer email, review snippet, or support message..."
                maxLength={500}
                className="mt-3 w-full rounded-2xl border border-slate-200 bg-slate-50/50 p-4 text-sm leading-relaxed text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 dark:border-slate-800 dark:bg-slate-950/60 dark:text-white dark:focus:bg-slate-950"
              />

              {/* Sample Ticket Shortcuts */}
              <div className="mt-4">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-2">
                  Pre-configured Ticket Scenarios:
                </span>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {sampleTickets.map((sample) => (
                    <button
                      key={sample.label}
                      type="button"
                      onClick={() => setText(sample.text)}
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
                  <strong>Evaluation Scope:</strong> This model is fine-tuned on the
                  SST-2 movie review corpus. Predictions serve as a decision-support
                  signal. Deployments on enterprise support tickets require domain adaptation.
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Output & Token Decomposition (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            {/* BERT Output Card */}
            {(model === "bert" || model === "both") && result.bert && (
              <div className="rounded-3xl border border-indigo-200/90 bg-white p-6 shadow-sm dark:border-indigo-900/60 dark:bg-slate-900">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3 dark:border-slate-800">
                  <div className="flex items-center gap-2">
                    <Cpu className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-200">
                      BERT Transformer Output
                    </span>
                  </div>
                  <span className="flex items-center gap-1 font-mono text-[11px] text-slate-400">
                    <Clock className="h-3 w-3" />
                    <span>{result.bert.latencyMs}ms</span>
                  </span>
                </div>

                <div className="mt-4 flex items-center justify-between">
                  <div>
                    <span className="text-[11px] uppercase font-bold text-slate-400">
                      Predicted Class
                    </span>
                    <div className="mt-1 flex items-center gap-2">
                      {result.bert.sentiment === "POSITIVE" ? (
                        <CheckCircle className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                      ) : (
                        <XCircle className="h-5 w-5 text-rose-600 dark:text-rose-400" />
                      )}
                      <span
                        className={`text-2xl font-black ${
                          result.bert.sentiment === "POSITIVE"
                            ? "text-emerald-700 dark:text-emerald-400"
                            : "text-rose-700 dark:text-rose-400"
                        }`}
                      >
                        {result.bert.sentiment}
                      </span>
                    </div>
                  </div>

                  <div className="text-right">
                    <span className="text-[11px] uppercase font-bold text-slate-400">
                      Confidence
                    </span>
                    <div className="mt-1 font-mono text-2xl font-black text-slate-900 dark:text-white">
                      {result.bert.confidencePercentage}%
                    </div>
                  </div>
                </div>

                {/* Meter */}
                <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      result.bert.sentiment === "POSITIVE"
                        ? "bg-emerald-500"
                        : "bg-rose-500"
                    }`}
                    style={{ width: `${result.bert.confidencePercentage}%` }}
                  />
                </div>

                {/* Token breakdown */}
                {result.bert.tokens && (
                  <div className="mt-5 border-t border-slate-100 pt-4 dark:border-slate-800">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-2">
                      WordPiece Sub-tokens:
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {result.bert.tokens.map((t, idx) => (
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
            )}

            {/* TF-IDF Output Card */}
            {(model === "tfidf" || model === "both") && result.tfidf && (
              <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3 dark:border-slate-800">
                  <div className="flex items-center gap-2">
                    <Binary className="h-4 w-4 text-slate-600 dark:text-slate-400" />
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-200">
                      TF-IDF + Linear SVM Output
                    </span>
                  </div>
                  <span className="flex items-center gap-1 font-mono text-[11px] text-slate-400">
                    <Clock className="h-3 w-3" />
                    <span>{result.tfidf.latencyMs}ms</span>
                  </span>
                </div>

                <div className="mt-4 flex items-center justify-between">
                  <div>
                    <span className="text-[11px] uppercase font-bold text-slate-400">
                      Predicted Class
                    </span>
                    <div className="mt-1 flex items-center gap-2">
                      {result.tfidf.sentiment === "POSITIVE" ? (
                        <CheckCircle className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                      ) : (
                        <XCircle className="h-5 w-5 text-rose-600 dark:text-rose-400" />
                      )}
                      <span
                        className={`text-2xl font-black ${
                          result.tfidf.sentiment === "POSITIVE"
                            ? "text-emerald-700 dark:text-emerald-400"
                            : "text-rose-700 dark:text-rose-400"
                        }`}
                      >
                        {result.tfidf.sentiment}
                      </span>
                    </div>
                  </div>

                  <div className="text-right">
                    <span className="text-[11px] uppercase font-bold text-slate-400">
                      Confidence
                    </span>
                    <div className="mt-1 font-mono text-2xl font-black text-slate-900 dark:text-white">
                      {result.tfidf.confidencePercentage}%
                    </div>
                  </div>
                </div>

                {/* Meter */}
                <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      result.tfidf.sentiment === "POSITIVE"
                        ? "bg-emerald-500"
                        : "bg-rose-500"
                    }`}
                    style={{ width: `${result.tfidf.confidencePercentage}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
