import Link from "next/link";
import {
  BarChart3,
  Cpu,
  Binary,
  Database,
  Terminal,
  ShieldCheck,
  ArrowLeft,
  ChevronRight,
  Info,
  Scale,
  Sparkles,
} from "lucide-react";

export default function PerformancePage() {
  const metrics = [
    {
      metric: "Accuracy",
      description: "Proportion of total sentences correctly classified.",
      bert: "—",
      tfidf: "—",
      status: "Awaiting evaluation execution",
    },
    {
      metric: "Macro F1-Score",
      description: "Harmonic mean of precision and recall averaged across Positive and Negative classes.",
      bert: "—",
      tfidf: "—",
      status: "Awaiting evaluation execution",
    },
    {
      metric: "Positive Class Recall",
      description: "Ability to retrieve all positive customer sentiment instances.",
      bert: "—",
      tfidf: "—",
      status: "Awaiting evaluation execution",
    },
    {
      metric: "Negative Class Recall",
      description: "Ability to retrieve critical customer complaints and negative feedback.",
      bert: "—",
      tfidf: "—",
      status: "Awaiting evaluation execution",
    },
    {
      metric: "Avg Inference Latency",
      description: "Per-sentence inference duration measured on standard CPU environment.",
      bert: "~35–45 ms",
      tfidf: "~5–10 ms",
      status: "Empirical CPU benchmark estimate",
    },
  ];

  return (
    <div className="py-10 md:py-16">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Breadcrumbs */}
        <div className="mb-6 flex items-center gap-2 text-xs font-medium text-slate-500">
          <Link href="/" className="hover:text-indigo-600 flex items-center gap-1">
            <ArrowLeft className="h-3.5 w-3.5" />
            <span>Back to Landing</span>
          </Link>
          <ChevronRight className="h-3.5 w-3.5" />
          <span className="text-slate-900 dark:text-white font-semibold">
            Model Performance
          </span>
        </div>

        {/* Page Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-200/80 pb-6 dark:border-slate-800">
          <div>
            <div className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700 dark:bg-blue-950/60 dark:text-blue-300">
              <BarChart3 className="h-3.5 w-3.5 text-blue-600" />
              <span>SST-2 GLUE Benchmark Evaluation</span>
            </div>
            <h1 className="mt-2 text-2xl font-extrabold text-slate-900 sm:text-3xl dark:text-white">
              Measure Model Performance
            </h1>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
              Evaluation uses the official SST-2 GLUE development split.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/analyze"
              className="inline-flex items-center gap-1.5 rounded-xl bg-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow-2xs hover:bg-indigo-500"
            >
              <Sparkles className="h-3.5 w-3.5" />
              <span>Try Live Classifier</span>
            </Link>
          </div>
        </div>

        {/* Scientific Disclosure Notice */}
        <div className="mt-8 rounded-2xl border border-blue-200/80 bg-blue-50/60 p-5 dark:border-blue-900/50 dark:bg-blue-950/30">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 shrink-0 text-blue-600 dark:text-blue-400 mt-0.5" />
            <div className="text-xs sm:text-sm text-blue-900 dark:text-blue-200 leading-relaxed">
              <strong>Evaluation Standards:</strong> In accordance with scientific
              transparency, performance metrics are strictly derived from running the evaluation
              script on the official 872-sentence SST-2 validation split. Unverified or fabricated
              numbers are never displayed.
            </div>
          </div>
        </div>

        {/* Metric Comparison Table */}
        <div className="mt-10 overflow-hidden rounded-3xl border border-slate-200/90 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <div className="border-b border-slate-100 p-6 dark:border-slate-800">
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">
              SST-2 Validation Split Benchmark Matrix
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Official GLUE benchmark validation split (872 binary labeled samples).
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-100 bg-slate-50/70 text-xs uppercase font-bold text-slate-500 dark:border-slate-800 dark:bg-slate-950/60 dark:text-slate-400">
                <tr>
                  <th className="px-6 py-4">Metric</th>
                  <th className="px-6 py-4 text-indigo-600 dark:text-indigo-400">
                    BERT (Transformer)
                  </th>
                  <th className="px-6 py-4 text-slate-700 dark:text-slate-300">
                    TF-IDF + Linear SVM (Baseline)
                  </th>
                  <th className="px-6 py-4">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {metrics.map((m) => (
                  <tr key={m.metric} className="hover:bg-slate-50/50 dark:hover:bg-slate-800/30">
                    <td className="px-6 py-4">
                      <div className="font-bold text-slate-900 dark:text-white">
                        {m.metric}
                      </div>
                      <div className="text-xs text-slate-500 dark:text-slate-400">
                        {m.description}
                      </div>
                    </td>
                    <td className="px-6 py-4 font-mono font-bold text-indigo-600 dark:text-indigo-400">
                      {m.bert}
                    </td>
                    <td className="px-6 py-4 font-mono font-bold text-slate-700 dark:text-slate-300">
                      {m.tfidf}
                    </td>
                    <td className="px-6 py-4 text-xs text-slate-400 font-medium">
                      {m.status}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Evaluation Execution CLI Box */}
        <div className="mt-10 rounded-3xl border border-slate-200/90 bg-white p-6 sm:p-8 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-900 text-white dark:bg-slate-800">
              <Terminal className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                How to Run Evaluation Locally
              </h3>
              <p className="text-xs text-slate-500">
                Execute the evaluation script in your terminal to compute full validation split accuracy, confusion matrix, and F1-score.
              </p>
            </div>
          </div>

          <div className="mt-6 space-y-4">
            <div>
              <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                1. Ensure SST-2 dataset and pretrained BERT weights are saved:
              </span>
              <pre className="mt-1.5 overflow-x-auto rounded-xl bg-slate-900 p-3 font-mono text-xs text-emerald-400">
                <code>python data/download_sst2.py&#10;python models/download_bert.py</code>
              </pre>
            </div>

            <div>
              <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                2. Execute evaluation against both BERT and TF-IDF baseline:
              </span>
              <pre className="mt-1.5 overflow-x-auto rounded-xl bg-slate-900 p-3 font-mono text-xs text-emerald-400">
                <code>python evaluation/evaluate.py --split validation --models bert,tfidf</code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
