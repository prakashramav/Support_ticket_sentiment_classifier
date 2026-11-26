import Link from "next/link";
import {
  BarChart3,
  TrendingUp,
  Terminal,
  ArrowRight,
  Info,
  CheckCircle,
  Database,
  CheckCircle2,
} from "lucide-react";

export default function PerformancePreviewSection() {
  return (
    <section className="border-t border-slate-200/80 bg-slate-50/60 py-20 dark:border-slate-800/80 dark:bg-slate-950/60 md:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="mx-auto max-w-3xl text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-indigo-600 dark:text-indigo-400">
            Evaluation Benchmarks
          </span>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
            Measured Model Performance
          </h2>
          <p className="mt-4 text-base text-slate-600 dark:text-slate-300">
            Empirical benchmark metrics evaluated on the official SST-2 GLUE validation split (N = 872).
          </p>
        </div>

        {/* Dashboard Preview Cards */}
        <div className="mt-14 mx-auto max-w-4xl">
          <div className="rounded-3xl border border-slate-200/90 bg-white p-6 sm:p-8 shadow-sm dark:border-slate-800 dark:bg-slate-900">
            {/* Top Bar with Split Details */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-100 pb-5 dark:border-slate-800">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600 dark:bg-indigo-950/60 dark:text-indigo-400">
                  <Database className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                    Official SST-2 Validation Split
                  </h3>
                  <span className="text-xs text-slate-400 font-mono">
                    872 validation sentences (GLUE Benchmark: 444 Pos, 428 Neg)
                  </span>
                </div>
              </div>

              <div className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300">
                <CheckCircle2 className="h-3.5 w-3.5" />
                <span>Empirical Evaluation Verified</span>
              </div>
            </div>

            {/* Metrics Comparison Grid */}
            <div className="mt-8 grid grid-cols-1 gap-6 md:grid-cols-2">
              {/* Metric 1: Accuracy */}
              <div className="rounded-2xl border border-slate-200/80 bg-slate-50/50 p-5 dark:border-slate-800 dark:bg-slate-950/50">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    Overall Accuracy
                  </span>
                  <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400">
                    +10.32% &Delta;
                  </span>
                </div>

                <div className="mt-6 space-y-4">
                  {/* BERT row */}
                  <div>
                    <div className="flex items-center justify-between text-xs font-medium">
                      <span className="font-bold text-indigo-600 dark:text-indigo-400">
                        BERT (Primary)
                      </span>
                      <span className="font-mono font-bold text-indigo-600 dark:text-indigo-400">92.89%</span>
                    </div>
                    <div className="mt-1.5 h-2.5 w-full rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div className="h-full rounded-full bg-indigo-600" style={{ width: "92.89%" }} />
                    </div>
                  </div>

                  {/* TF-IDF row */}
                  <div>
                    <div className="flex items-center justify-between text-xs font-medium">
                      <span className="font-bold text-slate-700 dark:text-slate-300">
                        TF-IDF + Logistic Regression
                      </span>
                      <span className="font-mono font-bold text-slate-600 dark:text-slate-400">82.57%</span>
                    </div>
                    <div className="mt-1.5 h-2.5 w-full rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div className="h-full rounded-full bg-slate-400 dark:bg-slate-500" style={{ width: "82.57%" }} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Metric 2: F1-Score */}
              <div className="rounded-2xl border border-slate-200/80 bg-slate-50/50 p-5 dark:border-slate-800 dark:bg-slate-950/50">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    Positive F1-Score
                  </span>
                  <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400">
                    +10.18% &Delta;
                  </span>
                </div>

                <div className="mt-6 space-y-4">
                  {/* BERT row */}
                  <div>
                    <div className="flex items-center justify-between text-xs font-medium">
                      <span className="font-bold text-indigo-600 dark:text-indigo-400">
                        BERT (Primary)
                      </span>
                      <span className="font-mono font-bold text-indigo-600 dark:text-indigo-400">93.06%</span>
                    </div>
                    <div className="mt-1.5 h-2.5 w-full rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div className="h-full rounded-full bg-indigo-600" style={{ width: "93.06%" }} />
                    </div>
                  </div>

                  {/* TF-IDF row */}
                  <div>
                    <div className="flex items-center justify-between text-xs font-medium">
                      <span className="font-bold text-slate-700 dark:text-slate-300">
                        TF-IDF + Logistic Regression
                      </span>
                      <span className="font-mono font-bold text-slate-600 dark:text-slate-400">82.88%</span>
                    </div>
                    <div className="mt-1.5 h-2.5 w-full rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div className="h-full rounded-full bg-slate-400 dark:bg-slate-500" style={{ width: "82.88%" }} />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Evaluation Instructions Box */}
            <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-950/80">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 dark:text-slate-300">
                  <Terminal className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
                  <span>Reproduce benchmark results locally:</span>
                </div>
                <span className="text-[10px] text-slate-400 font-mono">CLI Command</span>
              </div>
              <pre className="mt-2.5 overflow-x-auto rounded-lg bg-slate-900 p-3 font-mono text-xs text-emerald-400">
                <code>python evaluation/evaluate.py --split validation --models bert,tfidf</code>
              </pre>
            </div>

            {/* Bottom Link to /performance */}
            <div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-slate-100 pt-5 dark:border-slate-800">
              <span className="text-xs text-slate-500 dark:text-slate-400">
                Want to see confusion matrices, per-class recall, and false-positive rates?
              </span>
              <Link
                href="/performance"
                className="inline-flex items-center gap-1.5 text-xs font-bold text-indigo-600 hover:text-indigo-500 dark:text-indigo-400"
              >
                <span>Open Full Performance Dashboard</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
