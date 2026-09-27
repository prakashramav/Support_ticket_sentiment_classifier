import Link from "next/link";
import {
  Cpu,
  Binary,
  CheckCircle2,
  ArrowRight,
  Sparkles,
  Scale,
  Zap,
  Clock,
  Layers,
} from "lucide-react";

export default function ComparisonSection() {
  const bertPoints = [
    { title: "Context-aware", desc: "Captures word order, subtle negations, and syntax nuance." },
    { title: "Pretrained Transformer", desc: "Initialized from bert-base-uncased (110M parameters)." },
    { title: "Fine-tuned for sentiment", desc: "Classification head tuned on SST-2 binary sentences." },
    { title: "Primary model", desc: "Main decision engine for rich text understanding." },
    { title: "More computationally intensive", desc: "Requires matrix tensor multiplications; ideal for GPU or optimized CPU." },
  ];

  const tfidfPoints = [
    { title: "Classical NLP approach", desc: "Well-established bag-of-words statistical frequency baseline." },
    { title: "Numerical word features", desc: "Term Frequency × Inverse Document Frequency vector weights." },
    { title: "Lightweight", desc: "Extremely fast inference footprint with minimal memory requirements." },
    { title: "Baseline model", desc: "Provides an empirical lower-bound benchmark to evaluate transformer gains." },
    { title: "Easier to train and deploy", desc: "Trains in seconds on CPU without specialized hardware." },
  ];

  return (
    <section className="py-20 md:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="mx-auto max-w-3xl text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-indigo-600 dark:text-indigo-400">
            Architectural Trade-offs
          </span>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
            Modern Transformer vs Classical NLP
          </h2>
          <p className="mt-4 text-base text-slate-600 dark:text-slate-300">
            Compare both approaches using the same evaluation dataset and metrics.
          </p>
        </div>

        {/* Dual Comparison Cards */}
        <div className="mt-14 grid grid-cols-1 gap-8 lg:grid-cols-2">
          {/* BERT Card */}
          <div className="relative flex flex-col justify-between rounded-3xl border-2 border-indigo-500/80 bg-white p-8 shadow-lg shadow-indigo-500/5 dark:border-indigo-500/60 dark:bg-slate-900">
            <div className="absolute -top-3.5 right-6 rounded-full bg-indigo-600 px-3 py-1 text-xs font-bold text-white shadow-sm">
              Primary Model
            </div>

            <div>
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-50 text-indigo-600 dark:bg-indigo-950/60 dark:text-indigo-400">
                  <Cpu className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                    BERT (Transformer)
                  </h3>
                  <span className="text-xs text-indigo-600 dark:text-indigo-400 font-mono">
                    bert-base-uncased fine-tuned
                  </span>
                </div>
              </div>

              <ul className="mt-8 space-y-4">
                {bertPoints.map((point) => (
                  <li key={point.title} className="flex items-start gap-3">
                    <CheckCircle2 className="h-5 w-5 shrink-0 text-indigo-600 dark:text-indigo-400 mt-0.5" />
                    <div>
                      <span className="text-sm font-bold text-slate-800 dark:text-slate-100">
                        {point.title}:
                      </span>{" "}
                      <span className="text-sm text-slate-600 dark:text-slate-400">
                        {point.desc}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>

            <div className="mt-8 rounded-2xl bg-indigo-50/70 p-4 dark:bg-indigo-950/40 border border-indigo-100 dark:border-indigo-900/50">
              <span className="text-xs font-semibold text-indigo-900 dark:text-indigo-300">
                Primary Strength:
              </span>
              <p className="mt-1 text-xs text-indigo-800/80 dark:text-indigo-300/80 leading-relaxed">
                Excels at capturing contextual meaning, double negatives (&quot;not bad&quot;),
                and nuanced semantic shifts across multi-clause sentences.
              </p>
            </div>
          </div>

          {/* TF-IDF + Linear SVM Card */}
          <div className="flex flex-col justify-between rounded-3xl border border-slate-200 bg-white p-8 shadow-xs dark:border-slate-800 dark:bg-slate-900">
            <div>
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                  <Binary className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                    TF-IDF + Linear SVM
                  </h3>
                  <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                    scikit-learn baseline pipeline
                  </span>
                </div>
              </div>

              <ul className="mt-8 space-y-4">
                {tfidfPoints.map((point) => (
                  <li key={point.title} className="flex items-start gap-3">
                    <CheckCircle2 className="h-5 w-5 shrink-0 text-slate-500 dark:text-slate-400 mt-0.5" />
                    <div>
                      <span className="text-sm font-bold text-slate-800 dark:text-slate-100">
                        {point.title}:
                      </span>{" "}
                      <span className="text-sm text-slate-600 dark:text-slate-400">
                        {point.desc}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>

            <div className="mt-8 rounded-2xl bg-slate-50 p-4 dark:bg-slate-800/50 border border-slate-200/60 dark:border-slate-800">
              <span className="text-xs font-semibold text-slate-900 dark:text-slate-200">
                Primary Strength:
              </span>
              <p className="mt-1 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                Superb latency (sub-10ms), zero GPU dependency, interpretable feature coefficients,
                and a strong baseline for text with distinct lexical polarities.
              </p>
            </div>
          </div>
        </div>

        {/* Balanced Evaluation Principle Banner & CTA */}
        <div className="mt-10 flex flex-col sm:flex-row items-center justify-between gap-6 rounded-2xl border border-slate-200 bg-slate-50/80 p-6 dark:border-slate-800 dark:bg-slate-950/60">
          <div className="flex items-center gap-3">
            <Scale className="h-6 w-6 text-indigo-600 dark:text-indigo-400 shrink-0" />
            <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
              Rigorous science avoids assumptions. We benchmark both architectures across
              identical splits to substantiate real-world trade-offs.
            </p>
          </div>
          <Link
            href="/performance"
            className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition-all hover:bg-indigo-500"
          >
            <span>View Performance</span>
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </div>
    </section>
  );
}
