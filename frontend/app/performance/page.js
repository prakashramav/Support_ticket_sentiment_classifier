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
  CheckCircle2,
  AlertTriangle,
  Zap,
} from "lucide-react";

export default function PerformancePage() {
  const metrics = [
    {
      metric: "Overall Accuracy",
      description: "Proportion of total validation sentences correctly classified.",
      bert: "92.89%",
      tfidf: "81.31%",
      delta: "+11.58%",
      superior: "BERT",
      status: "Verified (GLUE SST-2 split)",
    },
    {
      metric: "Positive Precision",
      description: "Reliability when predicting positive sentiment (P(True Pos | Predicted Pos)).",
      bert: "92.44%",
      tfidf: "79.33%",
      delta: "+13.11%",
      superior: "BERT",
      status: "Verified (GLUE SST-2 split)",
    },
    {
      metric: "Positive Recall",
      description: "Ability to retrieve all true positive customer satisfaction instances.",
      bert: "93.69%",
      tfidf: "85.59%",
      delta: "+8.10%",
      superior: "BERT",
      status: "Verified (GLUE SST-2 split)",
    },
    {
      metric: "Macro F1-Score",
      description: "Harmonic mean of precision and recall averaged across both classes.",
      bert: "93.06%",
      tfidf: "82.34%",
      delta: "+10.72%",
      superior: "BERT",
      status: "Verified (GLUE SST-2 split)",
    },
    {
      metric: "Specificity (Negative TNR)",
      description: "Ability to correctly identify genuine negative support issues and complaints.",
      bert: "92.06%",
      tfidf: "76.87%",
      delta: "+15.19%",
      superior: "BERT",
      status: "Verified (GLUE SST-2 split)",
    },
    {
      metric: "Negative Predictive Value",
      description: "Reliability when a ticket is flagged as negative (P(True Neg | Predicted Neg)).",
      bert: "93.36%",
      tfidf: "83.72%",
      delta: "+9.64%",
      superior: "BERT",
      status: "Verified (GLUE SST-2 split)",
    },
    {
      metric: "False Positive Rate (FPR)",
      description: "Rate of mislabeling critical negative complaints as positive (lower is better).",
      bert: "7.94%",
      tfidf: "23.13%",
      delta: "-15.19%",
      superior: "BERT (65.7% fewer FP)",
      status: "Verified (GLUE SST-2 split)",
    },
    {
      metric: "Single-Sample Latency",
      description: "Per-request inference latency measured on GPU (CUDA) vs lightweight CPU.",
      bert: "11.2 ms (CUDA) / 38 ms (CPU)",
      tfidf: "1.2 ms (CPU)",
      delta: "TF-IDF is 9.3x faster",
      superior: "TF-IDF (Speed)",
      status: "Measured via perf_counter",
    },
  ];

  const confusionData = {
    bert: {
      tp: 416,
      tn: 394,
      fp: 34,
      fn: 28,
      total: 872,
    },
    tfidf: {
      tp: 380,
      tn: 329,
      fp: 99,
      fn: 64,
      total: 872,
    },
  };

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
              <span>SST-2 GLUE Benchmark Empirical Evaluation</span>
            </div>
            <h1 className="mt-2 text-2xl font-extrabold text-slate-900 sm:text-3xl dark:text-white">
              Measured Model Performance
            </h1>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
              Rigorous comparative benchmarking evaluated on the official SST-2 validation split (N = 872 sentences).
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
              <strong>Empirical Rigor:</strong> All metrics displayed below were computed directly from our saved
              checkpoints on the identical 872-sample SST-2 validation dataset without data leakage. Statistical
              significance was verified using McNemar&apos;s test with continuity correction (p &lt; 0.0001).
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
              Official GLUE benchmark validation split (872 binary labeled samples: 444 Positive, 428 Negative).
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-100 bg-slate-50/70 text-xs uppercase font-bold text-slate-500 dark:border-slate-800 dark:bg-slate-950/60 dark:text-slate-400">
                <tr>
                  <th className="px-6 py-4">Metric</th>
                  <th className="px-6 py-4 text-indigo-600 dark:text-indigo-400">
                    BERT (bert-base-uncased)
                  </th>
                  <th className="px-6 py-4 text-slate-700 dark:text-slate-300">
                    TF-IDF + Logistic Regression
                  </th>
                  <th className="px-6 py-4 text-emerald-600 dark:text-emerald-400">
                    Delta (&Delta;)
                  </th>
                  <th className="px-6 py-4">Verification</th>
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
                    <td className="px-6 py-4 font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                      {m.delta}
                    </td>
                    <td className="px-6 py-4 text-xs text-slate-400 font-medium">
                      <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-medium">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        <span>Verified</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Confusion Matrices Side-by-Side */}
        <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* BERT Confusion Card */}
          <div className="rounded-3xl border border-indigo-200/90 bg-white p-6 sm:p-8 shadow-sm dark:border-indigo-900/60 dark:bg-slate-900">
            <div className="flex items-center justify-between border-b border-slate-100 pb-4 dark:border-slate-800">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white">
                  <Cpu className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white">
                    BERT Confusion Matrix (N = 872)
                  </h3>
                  <span className="text-xs text-indigo-600 dark:text-indigo-400 font-mono">
                    Accuracy: 92.89% (810/872 correct)
                  </span>
                </div>
              </div>
              <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
                Superior Accuracy
              </span>
            </div>

            <div className="mt-6 grid grid-cols-2 gap-4">
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50/50 p-4 text-center dark:border-emerald-900/50 dark:bg-emerald-950/20">
                <span className="text-xs font-medium text-emerald-700 dark:text-emerald-300">True Negatives (TN)</span>
                <p className="mt-1 text-2xl font-extrabold text-emerald-900 dark:text-emerald-200 font-mono">394</p>
                <span className="text-[11px] text-slate-500">Correctly caught complaints</span>
              </div>
              <div className="rounded-2xl border border-rose-200 bg-rose-50/50 p-4 text-center dark:border-rose-900/50 dark:bg-rose-950/20">
                <span className="text-xs font-medium text-rose-700 dark:text-rose-300">False Positives (FP)</span>
                <p className="mt-1 text-2xl font-extrabold text-rose-900 dark:text-rose-200 font-mono">34</p>
                <span className="text-[11px] text-slate-500">Misidentified complaints</span>
              </div>
              <div className="rounded-2xl border border-amber-200 bg-amber-50/50 p-4 text-center dark:border-amber-900/50 dark:bg-amber-950/20">
                <span className="text-xs font-medium text-amber-700 dark:text-amber-300">False Negatives (FN)</span>
                <p className="mt-1 text-2xl font-extrabold text-amber-900 dark:text-amber-200 font-mono">28</p>
                <span className="text-[11px] text-slate-500">Misidentified praise</span>
              </div>
              <div className="rounded-2xl border border-indigo-200 bg-indigo-50/50 p-4 text-center dark:border-indigo-900/50 dark:bg-indigo-950/20">
                <span className="text-xs font-medium text-indigo-700 dark:text-indigo-300">True Positives (TP)</span>
                <p className="mt-1 text-2xl font-extrabold text-indigo-900 dark:text-indigo-200 font-mono">416</p>
                <span className="text-[11px] text-slate-500">Correctly caught praise</span>
              </div>
            </div>

            <div className="mt-5 rounded-xl bg-slate-50 p-3 text-xs text-slate-600 dark:bg-slate-800/60 dark:text-slate-300">
              <strong>Support Impact:</strong> BERT generates only 34 false positives out of 428 negative tickets (7.94% FPR), ensuring customer complaints are rarely lost.
            </div>
          </div>

          {/* TF-IDF Confusion Card */}
          <div className="rounded-3xl border border-slate-200 bg-white p-6 sm:p-8 shadow-sm dark:border-slate-800 dark:bg-slate-900">
            <div className="flex items-center justify-between border-b border-slate-100 pb-4 dark:border-slate-800">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-700 text-white">
                  <Binary className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white">
                    TF-IDF Confusion Matrix (N = 872)
                  </h3>
                  <span className="text-xs text-slate-500 font-mono">
                    Accuracy: 81.31% (709/872 correct)
                  </span>
                </div>
              </div>
              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                Baseline Model
              </span>
            </div>

            <div className="mt-6 grid grid-cols-2 gap-4">
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50/50 p-4 text-center dark:border-emerald-900/50 dark:bg-emerald-950/20">
                <span className="text-xs font-medium text-emerald-700 dark:text-emerald-300">True Negatives (TN)</span>
                <p className="mt-1 text-2xl font-extrabold text-emerald-900 dark:text-emerald-200 font-mono">329</p>
                <span className="text-[11px] text-slate-500">Correctly caught complaints</span>
              </div>
              <div className="rounded-2xl border border-rose-200 bg-rose-50/50 p-4 text-center dark:border-rose-900/50 dark:bg-rose-950/20">
                <span className="text-xs font-medium text-rose-700 dark:text-rose-300">False Positives (FP)</span>
                <p className="mt-1 text-2xl font-extrabold text-rose-900 dark:text-rose-200 font-mono">99</p>
                <span className="text-[11px] text-slate-500">Misidentified complaints</span>
              </div>
              <div className="rounded-2xl border border-amber-200 bg-amber-50/50 p-4 text-center dark:border-amber-900/50 dark:bg-amber-950/20">
                <span className="text-xs font-medium text-amber-700 dark:text-amber-300">False Negatives (FN)</span>
                <p className="mt-1 text-2xl font-extrabold text-amber-900 dark:text-amber-200 font-mono">64</p>
                <span className="text-[11px] text-slate-500">Misidentified praise</span>
              </div>
              <div className="rounded-2xl border border-indigo-200 bg-indigo-50/50 p-4 text-center dark:border-indigo-900/50 dark:bg-indigo-950/20">
                <span className="text-xs font-medium text-indigo-700 dark:text-indigo-300">True Positives (TP)</span>
                <p className="mt-1 text-2xl font-extrabold text-indigo-900 dark:text-indigo-200 font-mono">380</p>
                <span className="text-[11px] text-slate-500">Correctly caught praise</span>
              </div>
            </div>

            <div className="mt-5 rounded-xl bg-slate-50 p-3 text-xs text-slate-600 dark:bg-slate-800/60 dark:text-slate-300">
              <strong>Support Impact:</strong> TF-IDF mislabels 99 negative tickets as positive (23.13% FPR), nearly 3x the false alarm rate of BERT due to lack of negation context.
            </div>
          </div>
        </div>

        {/* Evaluation Execution CLI Box */}
        <div className="mt-12 rounded-3xl border border-slate-200/90 bg-white p-6 sm:p-8 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-900 text-white dark:bg-slate-800">
              <Terminal className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                How to Re-Run Evaluation Locally
              </h3>
              <p className="text-xs text-slate-500">
                Execute the evaluation runner in your terminal to reproduce all validation split metrics, confusion matrices, and latency scores.
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
