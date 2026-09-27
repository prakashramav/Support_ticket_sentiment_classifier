"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Sparkles,
  ArrowRight,
  Cpu,
  CheckCircle2,
  Layers,
  ArrowUpRight,
  Activity,
} from "lucide-react";

export default function HeroSection() {
  const [activeStep, setActiveStep] = useState(3); // text -> tokenize -> bert -> prediction

  const pipelineSteps = [
    { name: "Text Input", label: "Raw Customer Text" },
    { name: "Tokenization", label: "WordPiece (30,522 Vocab)" },
    { name: "BERT Encoder", label: "12-Layer Transformer" },
    { name: "Prediction", label: "Softmax Classification" },
  ];

  return (
    <section className="relative overflow-hidden pt-12 pb-20 md:pt-20 md:pb-28">
      {/* Subtle ambient lighting behind hero */}
      <div
        className="pointer-events-none absolute -top-24 left-1/2 -z-10 h-[500px] w-[800px] -translate-x-1/2 rounded-full bg-gradient-to-tr from-indigo-500/15 via-violet-500/10 to-sky-500/10 blur-3xl dark:from-indigo-600/20 dark:via-purple-600/15 dark:to-cyan-600/10"
        aria-hidden="true"
      />

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-12 lg:gap-8">
          {/* Left Column: Heading & CTAs */}
          <div className="lg:col-span-7 flex flex-col items-center text-center lg:items-start lg:text-left">
            {/* Small Badge */}
            <div className="inline-flex items-center gap-2 rounded-full border border-indigo-200/80 bg-indigo-50/80 px-3.5 py-1 text-xs font-semibold text-indigo-700 shadow-2xs backdrop-blur-xs transition-all hover:bg-indigo-100/80 dark:border-indigo-800/80 dark:bg-indigo-950/70 dark:text-indigo-300">
              <Sparkles className="h-3.5 w-3.5 text-indigo-600 dark:text-indigo-400" />
              <span>AI-Powered Sentiment Analysis</span>
              <span className="h-1 w-1 rounded-full bg-indigo-400" />
              <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                BERT + TF-IDF Baseline
              </span>
            </div>

            {/* Main Heading */}
            <h1 className="mt-5 text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl dark:text-white">
              Understand Customer Sentiment{" "}
              <span className="bg-gradient-to-r from-indigo-600 via-violet-600 to-indigo-500 bg-clip-text text-transparent dark:from-indigo-400 dark:via-violet-400 dark:to-cyan-300">
                Before It Becomes
              </span>{" "}
              a Problem.
            </h1>

            {/* Supporting Text */}
            <p className="mt-6 max-w-2xl text-lg leading-relaxed text-slate-600 dark:text-slate-300">
              Analyze text instantly with a fine-tuned BERT sentiment classifier
              and compare its predictions with a traditional TF-IDF
              machine-learning baseline.
            </p>

            {/* CTAs */}
            <div className="mt-8 flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto">
              <Link
                href="/analyze"
                className="group w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-6 py-3.5 text-base font-semibold text-white shadow-md shadow-indigo-600/25 transition-all hover:shadow-lg hover:shadow-indigo-600/35 hover:from-indigo-500 hover:to-violet-500 active:scale-[0.98]"
              >
                <span>Analyze Sentiment</span>
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Link>
              <Link
                href="/performance"
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white/80 px-6 py-3.5 text-base font-semibold text-slate-800 shadow-2xs backdrop-blur-xs transition-all hover:bg-slate-50 hover:border-slate-400 dark:border-slate-700 dark:bg-slate-900/80 dark:text-slate-200 dark:hover:bg-slate-800"
              >
                <span>View Performance</span>
                <ArrowUpRight className="h-4 w-4 text-slate-400" />
              </Link>
            </div>

            {/* Micro proof note */}
            <div className="mt-8 flex flex-wrap items-center justify-center lg:justify-start gap-4 text-xs text-slate-500 dark:text-slate-400">
              <div className="flex items-center gap-1.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                <span>Fine-tuned BERT-base</span>
              </div>
              <span className="hidden sm:inline text-slate-300 dark:text-slate-700">•</span>
              <div className="flex items-center gap-1.5">
                <CheckCircle2 className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
                <span>TF-IDF + Linear SVM Baseline</span>
              </div>
              <span className="hidden sm:inline text-slate-300 dark:text-slate-700">•</span>
              <div className="flex items-center gap-1.5">
                <CheckCircle2 className="h-4 w-4 text-sky-600 dark:text-sky-400" />
                <span>FastAPI Backend Ready</span>
              </div>
            </div>
          </div>

          {/* Right Column: Interactive Sentiment Analysis Preview Card */}
          <div className="lg:col-span-5 w-full">
            <div className="relative mx-auto max-w-lg rounded-3xl border border-slate-200/90 bg-white/95 p-6 shadow-xl shadow-indigo-500/5 ring-1 ring-slate-900/5 backdrop-blur-md dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-2xl dark:shadow-indigo-950/30">
              {/* Card Header */}
              <div className="flex items-center justify-between border-b border-slate-100 pb-4 dark:border-slate-800">
                <div className="flex items-center gap-2">
                  <span className="flex h-3 w-3 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                  </span>
                  <span className="text-xs font-semibold tracking-wide uppercase text-slate-500 dark:text-slate-400">
                    Live Classification Engine
                  </span>
                </div>
                <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-mono text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                  bert-sst2-v1
                </span>
              </div>

              {/* Sample Ticket Box */}
              <div className="mt-4 rounded-xl border border-slate-200/70 bg-slate-50/70 p-4 dark:border-slate-800 dark:bg-slate-950/60">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>Sample Customer Ticket</span>
                  <span className="font-mono text-[11px]">82 chars</span>
                </div>
                <p className="mt-2 text-sm font-medium leading-relaxed text-slate-800 dark:text-slate-200">
                  “I am really happy with the service. Your support team solved my
                  problem quickly.”
                </p>
              </div>

              {/* Subtle Animated Processing / Token Pipeline */}
              <div className="mt-5">
                <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">
                  <span>Pipeline Architecture</span>
                  <span className="text-indigo-600 dark:text-indigo-400">Inference Flow</span>
                </div>
                <div className="grid grid-cols-4 gap-1.5 text-center">
                  {pipelineSteps.map((step, idx) => (
                    <div
                      key={step.name}
                      className={`rounded-lg py-2 px-1 text-[11px] font-medium border transition-all ${
                        idx <= activeStep
                          ? "border-indigo-300 bg-indigo-50/80 text-indigo-700 dark:border-indigo-800 dark:bg-indigo-950/60 dark:text-indigo-300"
                          : "border-slate-200 bg-white text-slate-400 dark:border-slate-800 dark:bg-slate-900"
                      }`}
                    >
                      <div className="font-semibold">{step.name}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Prediction Result Grid */}
              <div className="mt-6 rounded-2xl border border-emerald-200/80 bg-emerald-50/60 p-4 dark:border-emerald-900/60 dark:bg-emerald-950/30">
                <div className="grid grid-cols-2 gap-4">
                  {/* Prediction */}
                  <div>
                    <span className="text-[11px] font-semibold tracking-wider text-emerald-800 uppercase dark:text-emerald-300">
                      Prediction
                    </span>
                    <div className="mt-1 flex items-center gap-1.5">
                      <span className="inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
                      <span className="text-xl font-black tracking-tight text-emerald-700 dark:text-emerald-300">
                        POSITIVE
                      </span>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-semibold tracking-wider text-emerald-800 uppercase dark:text-emerald-300">
                        Confidence
                      </span>
                      <span className="text-sm font-bold text-emerald-700 dark:text-emerald-300">
                        95%
                      </span>
                    </div>
                    {/* Confidence Meter */}
                    <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-emerald-200 dark:bg-emerald-900/60">
                      <div
                        className="h-full rounded-full bg-emerald-500 transition-all duration-1000"
                        style={{ width: "95%" }}
                      />
                    </div>
                  </div>
                </div>

                {/* Model & Version Details */}
                <div className="mt-4 flex items-center justify-between border-t border-emerald-200/60 pt-3 text-xs text-emerald-900 dark:border-emerald-900/50 dark:text-emerald-200">
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-500 dark:text-slate-400">Model:</span>
                    <span className="font-bold text-slate-800 dark:text-slate-100">BERT</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-500 dark:text-slate-400">Version:</span>
                    <span className="font-mono text-[11px] font-semibold text-slate-800 dark:text-slate-100">
                      bert-sst2-v1
                    </span>
                  </div>
                </div>
              </div>

              {/* Bottom Quick Action */}
              <div className="mt-4 text-center">
                <Link
                  href="/analyze"
                  className="inline-flex items-center gap-1 text-xs font-semibold text-indigo-600 transition-colors hover:text-indigo-500 dark:text-indigo-400"
                >
                  <span>Open Interactive Text Studio</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
