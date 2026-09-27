"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Sparkles,
  ArrowRight,
  Cpu,
  Binary,
  RotateCw,
  AlertCircle,
  CheckCircle,
  XCircle,
  HelpCircle,
  SlidersHorizontal,
} from "lucide-react";
import { predictSentiment } from "../services/api";

export default function LiveAnalyzerSection() {
  const [text, setText] = useState(
    "I am really happy with the service. Your support team solved my problem quickly."
  );
  const [selectedModel, setSelectedModel] = useState("bert"); // 'bert' | 'tfidf' | 'both'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState({
    sentiment: "POSITIVE",
    confidence: 0.94,
    confidencePercentage: 94,
    model: "BERT (bert-sst2-v1)",
    modelKey: "bert",
  });
  const [comparisonResult, setComparisonResult] = useState(null);

  const sampleTickets = [
    {
      label: "Positive Praise",
      text: "I am really happy with the service. Your support team solved my problem quickly.",
    },
    {
      label: "Negative Escalation",
      text: "This service is terrible. The system crashed twice and my data was completely lost.",
    },
    {
      label: "Feature Inconvenience",
      text: "The new update is frustrating and difficult to navigate. Please revert the changes.",
    },
  ];

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await predictSentiment(text, selectedModel);
      if (selectedModel === "both") {
        setComparisonResult(res);
        setResult(res.bert);
      } else {
        setComparisonResult(null);
        setResult(res);
      }
    } catch (err) {
      console.error("Inference error:", err);
    } finally {
      setLoading(false);
    }
  };

  const isPositive = result?.sentiment === "POSITIVE";

  return (
    <section id="try-analyzer" className="border-t border-slate-200/80 bg-slate-50/50 py-20 dark:border-slate-800/80 dark:bg-slate-950/50 md:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="mx-auto max-w-3xl text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-indigo-600 dark:text-indigo-400">
            Live Demonstration
          </span>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
            Try Sentiment Analysis
          </h2>
          <p className="mt-4 text-base text-slate-600 dark:text-slate-300">
            Test custom support inquiries or customer sentences directly. Compare
            how deep contextual embeddings evaluate tone compared to TF-IDF n-grams.
          </p>
        </div>

        {/* Interactive Analyzer Card */}
        <div className="mt-12 mx-auto max-w-4xl rounded-3xl border border-slate-200/90 bg-white shadow-xl shadow-indigo-500/5 dark:border-slate-800 dark:bg-slate-900">
          <div className="p-6 sm:p-8">
            {/* Model Selector Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-100 pb-6 dark:border-slate-800">
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Select Classification Model:
                </span>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setSelectedModel("bert")}
                    className={`inline-flex items-center gap-2 rounded-xl px-3.5 py-1.5 text-xs font-semibold transition-all ${
                      selectedModel === "bert"
                        ? "bg-indigo-600 text-white shadow-sm shadow-indigo-600/30"
                        : "bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300"
                    }`}
                  >
                    <span className="h-2 w-2 rounded-full bg-current" />
                    <span>BERT (Primary)</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setSelectedModel("tfidf")}
                    className={`inline-flex items-center gap-2 rounded-xl px-3.5 py-1.5 text-xs font-semibold transition-all ${
                      selectedModel === "tfidf"
                        ? "bg-indigo-600 text-white shadow-sm shadow-indigo-600/30"
                        : "bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300"
                    }`}
                  >
                    <span className="h-2 w-2 rounded-full bg-current" />
                    <span>TF-IDF (Baseline)</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setSelectedModel("both")}
                    className={`inline-flex items-center gap-2 rounded-xl px-3.5 py-1.5 text-xs font-semibold transition-all ${
                      selectedModel === "both"
                        ? "bg-indigo-600 text-white shadow-sm shadow-indigo-600/30"
                        : "bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300"
                    }`}
                  >
                    <span className="h-2 w-2 rounded-full bg-current" />
                    <span>Compare Both</span>
                  </button>
                </div>
              </div>

              {/* Sample Ticket Shortcuts */}
              <div className="flex flex-col items-start sm:items-end">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  Quick Sample Prompts:
                </span>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {sampleTickets.map((sample) => (
                    <button
                      key={sample.label}
                      type="button"
                      onClick={() => setText(sample.text)}
                      className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600 hover:border-indigo-300 hover:bg-white hover:text-indigo-600 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-400 dark:hover:text-indigo-400"
                    >
                      {sample.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Input Textarea with Character Counter */}
            <div className="mt-6">
              <label
                htmlFor="ticket-text"
                className="block text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-2"
              >
                Input Message:
              </label>
              <div className="relative">
                <textarea
                  id="ticket-text"
                  rows={3}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter a review or support message..."
                  className="w-full resize-none rounded-2xl border border-slate-200 bg-slate-50/50 p-4 text-sm leading-relaxed text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 dark:border-slate-800 dark:bg-slate-950/60 dark:text-white dark:focus:bg-slate-950"
                  maxLength={500}
                />
                <div className="absolute bottom-3 right-3 text-[11px] font-mono text-slate-400">
                  {text.length}/500 chars
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="mt-4 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                <HelpCircle className="h-3.5 w-3.5" />
                <span>Text will be tokenized and classified immediately.</span>
              </div>

              <button
                type="button"
                disabled={loading || !text.trim()}
                onClick={handleAnalyze}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-6 py-2.5 text-sm font-semibold text-white shadow-sm shadow-indigo-600/30 transition-all hover:shadow-indigo-600/40 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RotateCw className="h-4 w-4 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    <span>Analyze Sentiment</span>
                  </>
                )}
              </button>
            </div>

            {/* Result Panel */}
            <div className="mt-8 rounded-2xl border border-slate-200/90 bg-slate-50/70 p-5 dark:border-slate-800 dark:bg-slate-950/70">
              <div className="flex items-center justify-between border-b border-slate-200/60 pb-3 dark:border-slate-800">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Inference Results
                </span>
                <span className="rounded-full bg-white px-2.5 py-0.5 text-[11px] font-medium text-slate-600 border border-slate-200 dark:bg-slate-900 dark:border-slate-800 dark:text-slate-300">
                  {result?.model || "BERT"}
                </span>
              </div>

              {/* Normal Result or Dual Comparison */}
              {selectedModel === "both" && comparisonResult ? (
                <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* BERT Card */}
                  <div className="rounded-xl border border-indigo-200 bg-white p-4 dark:border-indigo-900 dark:bg-slate-900">
                    <div className="flex items-center justify-between text-xs font-bold text-indigo-700 dark:text-indigo-400">
                      <span>BERT Model</span>
                      <span className="font-mono text-[11px]">Primary</span>
                    </div>
                    <div className="mt-2 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {comparisonResult.bert.sentiment === "POSITIVE" ? (
                          <CheckCircle className="h-5 w-5 text-emerald-600" />
                        ) : (
                          <XCircle className="h-5 w-5 text-rose-600" />
                        )}
                        <span
                          className={`text-lg font-black ${
                            comparisonResult.bert.sentiment === "POSITIVE"
                              ? "text-emerald-700 dark:text-emerald-400"
                              : "text-rose-700 dark:text-rose-400"
                          }`}
                        >
                          {comparisonResult.bert.sentiment}
                        </span>
                      </div>
                      <span className="text-sm font-bold">
                        {comparisonResult.bert.confidencePercentage}%
                      </span>
                    </div>
                  </div>

                  {/* TF-IDF Card */}
                  <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
                    <div className="flex items-center justify-between text-xs font-bold text-slate-700 dark:text-slate-300">
                      <span>TF-IDF + Linear SVM</span>
                      <span className="font-mono text-[11px]">Baseline</span>
                    </div>
                    <div className="mt-2 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {comparisonResult.tfidf.sentiment === "POSITIVE" ? (
                          <CheckCircle className="h-5 w-5 text-emerald-600" />
                        ) : (
                          <XCircle className="h-5 w-5 text-rose-600" />
                        )}
                        <span
                          className={`text-lg font-black ${
                            comparisonResult.tfidf.sentiment === "POSITIVE"
                              ? "text-emerald-700 dark:text-emerald-400"
                              : "text-rose-700 dark:text-rose-400"
                          }`}
                        >
                          {comparisonResult.tfidf.sentiment}
                        </span>
                      </div>
                      <span className="text-sm font-bold">
                        {comparisonResult.tfidf.confidencePercentage}%
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-4 items-center">
                  {/* Sentiment Metric */}
                  <div className="rounded-xl border border-slate-200/80 bg-white p-3.5 dark:border-slate-800 dark:bg-slate-900">
                    <span className="text-[11px] font-semibold uppercase text-slate-400 tracking-wider">
                      Sentiment
                    </span>
                    <div className="mt-1 flex items-center gap-2">
                      {isPositive ? (
                        <CheckCircle className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                      ) : (
                        <XCircle className="h-5 w-5 text-rose-600 dark:text-rose-400" />
                      )}
                      <span
                        className={`text-xl font-black tracking-tight ${
                          isPositive
                            ? "text-emerald-700 dark:text-emerald-400"
                            : "text-rose-700 dark:text-rose-400"
                        }`}
                      >
                        {result?.sentiment || "POSITIVE"}
                      </span>
                    </div>
                  </div>

                  {/* Confidence Metric */}
                  <div className="rounded-xl border border-slate-200/80 bg-white p-3.5 dark:border-slate-800 dark:bg-slate-900">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-semibold uppercase text-slate-400 tracking-wider">
                        Confidence
                      </span>
                      <span
                        className={`text-sm font-black ${
                          isPositive
                            ? "text-emerald-700 dark:text-emerald-400"
                            : "text-rose-700 dark:text-rose-400"
                        }`}
                      >
                        {result?.confidencePercentage || 94}%
                      </span>
                    </div>
                    <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isPositive ? "bg-emerald-500" : "bg-rose-500"
                        }`}
                        style={{
                          width: `${result?.confidencePercentage || 94}%`,
                        }}
                      />
                    </div>
                  </div>

                  {/* Model Selected */}
                  <div className="rounded-xl border border-slate-200/80 bg-white p-3.5 dark:border-slate-800 dark:bg-slate-900">
                    <span className="text-[11px] font-semibold uppercase text-slate-400 tracking-wider">
                      Active Model
                    </span>
                    <div className="mt-1 flex items-center gap-1.5">
                      <Cpu className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
                      <span className="text-sm font-bold text-slate-800 dark:text-slate-100 truncate">
                        {result?.model || "BERT"}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Note: Prediction is a decision-support signal, not certainty */}
              <div className="mt-4 flex items-center gap-2 rounded-xl bg-amber-50 px-3.5 py-2.5 text-xs text-amber-900 dark:bg-amber-950/40 dark:text-amber-300 border border-amber-200/60 dark:border-amber-900/40">
                <AlertCircle className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
                <span>
                  <strong>Note:</strong> Prediction is a decision-support signal, not
                  certainty. Human review should remain part of ticket escalation.
                </span>
              </div>
            </div>

            {/* Encouragement Link to /analyze */}
            <div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-slate-100 pt-5 dark:border-slate-800">
              <span className="text-xs text-slate-500 dark:text-slate-400 text-center sm:text-left">
                Need token decomposition, logits inspection, and latency metrics?
              </span>
              <Link
                href="/analyze"
                className="inline-flex items-center gap-1.5 text-xs font-bold text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 transition-colors"
              >
                <span>Launch Full Text Analyzer Studio</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
