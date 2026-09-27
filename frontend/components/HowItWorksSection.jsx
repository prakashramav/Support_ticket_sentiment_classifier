import {
  FileText,
  Cpu,
  Binary,
  Layers,
  Sparkles,
  ArrowRight,
  TrendingUp,
  Percent,
} from "lucide-react";

export default function HowItWorksSection() {
  const steps = [
    {
      num: "01",
      title: "Enter Text",
      desc: "User submits support message or sentence.",
      icon: FileText,
    },
    {
      num: "02",
      title: "Tokenize / Vectorize",
      desc: "WordPiece tokens or TF-IDF N-gram counts.",
      icon: Binary,
    },
    {
      num: "03",
      title: "BERT or TF-IDF Model",
      desc: "Deep Transformer encoder or Linear SVM boundary.",
      icon: Cpu,
    },
    {
      num: "04",
      title: "Sentiment Prediction",
      desc: "Binary classification (Positive / Negative).",
      icon: Sparkles,
    },
    {
      num: "05",
      title: "Confidence Score",
      desc: "Calibrated probability of prediction reliability.",
      icon: Percent,
    },
  ];

  return (
    <section className="border-t border-slate-200/80 bg-slate-50/60 py-20 dark:border-slate-800/80 dark:bg-slate-950/60 md:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="mx-auto max-w-3xl text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-indigo-600 dark:text-indigo-400">
            Pipeline Architecture
          </span>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
            From Text to Sentiment in Seconds
          </h2>
          <p className="mt-4 text-base text-slate-600 dark:text-slate-300">
            A transparent 5-stage inference flow that vectorizes text, executes
            model inference, and outputs calibrated sentiment probabilities.
          </p>
        </div>

        {/* 5-Step Horizontal Workflow */}
        <div className="mt-14 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div
                key={step.num}
                className="relative flex flex-col justify-between rounded-2xl border border-slate-200/80 bg-white p-6 shadow-2xs transition-all hover:border-indigo-300 hover:shadow-md dark:border-slate-800 dark:bg-slate-900 dark:hover:border-indigo-700"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-2xl font-black text-indigo-600/80 dark:text-indigo-400/80">
                      {step.num}
                    </span>
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200">
                      <Icon className="h-5 w-5" />
                    </div>
                  </div>
                  <h3 className="mt-4 text-base font-bold text-slate-900 dark:text-white">
                    {step.title}
                  </h3>
                  <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                    {step.desc}
                  </p>
                </div>

                {/* Arrow connector indicator on desktop */}
                {idx < 4 && (
                  <div className="hidden lg:block absolute -right-3 top-1/2 -translate-y-1/2 z-10 text-slate-300 dark:text-slate-700">
                    <ArrowRight className="h-5 w-5" />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Dual Model Paths Visual */}
        <div className="mt-16 grid grid-cols-1 gap-8 lg:grid-cols-2">
          {/* BERT PATH */}
          <div className="rounded-3xl border border-indigo-200 bg-white p-7 shadow-sm dark:border-indigo-900/60 dark:bg-slate-900">
            <div className="flex items-center justify-between border-b border-slate-100 pb-4 dark:border-slate-800">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white">
                  <Cpu className="h-4 w-4" />
                </div>
                <div>
                  <h4 className="text-base font-bold text-slate-900 dark:text-white">
                    BERT Path (Primary Model)
                  </h4>
                  <span className="text-xs text-indigo-600 dark:text-indigo-400">
                    Pretrained Bidirectional Transformer
                  </span>
                </div>
              </div>
              <span className="rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-semibold text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
                Context-Aware
              </span>
            </div>

            <div className="mt-6 space-y-3">
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-xs font-bold text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
                  1
                </span>
                <span className="rounded-lg bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 border border-slate-200 dark:bg-slate-800/80 dark:border-slate-700 dark:text-slate-200 w-full">
                  Input Sentence / Ticket Text
                </span>
              </div>
              <div className="flex justify-center text-slate-300 dark:text-slate-700">↓</div>
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-xs font-bold text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
                  2
                </span>
                <span className="rounded-lg bg-indigo-50 px-3 py-2 text-xs font-medium text-indigo-800 border border-indigo-200 dark:bg-indigo-950/60 dark:border-indigo-800 dark:text-indigo-200 w-full">
                  BERT WordPiece Tokenizer (CLS, SEP, Vocab 30,522)
                </span>
              </div>
              <div className="flex justify-center text-slate-300 dark:text-slate-700">↓</div>
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-xs font-bold text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
                  3
                </span>
                <span className="rounded-lg bg-indigo-50 px-3 py-2 text-xs font-medium text-indigo-800 border border-indigo-200 dark:bg-indigo-950/60 dark:border-indigo-800 dark:text-indigo-200 w-full">
                  Fine-Tuned BERT Transformer Encoder (12 Layers, 768-dim)
                </span>
              </div>
              <div className="flex justify-center text-slate-300 dark:text-slate-700">↓</div>
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-xs font-bold text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
                  4
                </span>
                <span className="rounded-lg bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 border border-slate-200 dark:bg-slate-800/80 dark:border-slate-700 dark:text-slate-200 w-full">
                  Linear Classification Head + Softmax
                </span>
              </div>
              <div className="flex justify-center text-slate-300 dark:text-slate-700">↓</div>
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white">
                  5
                </span>
                <div className="flex w-full items-center justify-between rounded-lg bg-gradient-to-r from-emerald-50 to-teal-50 px-3 py-2 text-xs font-bold text-teal-800 border border-teal-200 dark:from-emerald-950/60 dark:to-teal-950/60 dark:border-teal-800 dark:text-teal-200">
                  <span>POSITIVE (1) or NEGATIVE (0)</span>
                  <span className="text-[11px] font-normal text-teal-700 dark:text-teal-300">
                    + Calibrated Probability
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* BASELINE PATH */}
          <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm dark:border-slate-800 dark:bg-slate-900">
            <div className="flex items-center justify-between border-b border-slate-100 pb-4 dark:border-slate-800">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-700 text-white dark:bg-slate-800">
                  <Binary className="h-4 w-4" />
                </div>
                <div>
                  <h4 className="text-base font-bold text-slate-900 dark:text-white">
                    Baseline Path (TF-IDF + SVM)
                  </h4>
                  <span className="text-xs text-slate-500 dark:text-slate-400">
                    Classical Machine Learning Pipeline
                  </span>
                </div>
              </div>
              <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-semibold text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                Lightweight Baseline
              </span>
            </div>

            <div className="mt-6 space-y-3">
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                  1
                </span>
                <span className="rounded-lg bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 border border-slate-200 dark:bg-slate-800/80 dark:border-slate-700 dark:text-slate-200 w-full">
                  Input Sentence / Ticket Text
                </span>
              </div>
              <div className="flex justify-center text-slate-300 dark:text-slate-700">↓</div>
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                  2
                </span>
                <span className="rounded-lg bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 border border-slate-200 dark:bg-slate-800/80 dark:border-slate-700 dark:text-slate-200 w-full">
                  TF-IDF Vectorizer (Term Frequency–Inverse Document Frequency)
                </span>
              </div>
              <div className="flex justify-center text-slate-300 dark:text-slate-700">↓</div>
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                  3
                </span>
                <span className="rounded-lg bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 border border-slate-200 dark:bg-slate-800/80 dark:border-slate-700 dark:text-slate-200 w-full">
                  Sparse Bag-of-Words Feature Matrix
                </span>
              </div>
              <div className="flex justify-center text-slate-300 dark:text-slate-700">↓</div>
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                  4
                </span>
                <span className="rounded-lg bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 border border-slate-200 dark:bg-slate-800/80 dark:border-slate-700 dark:text-slate-200 w-full">
                  Linear Support Vector Machine (Linear SVM) Hyperplane
                </span>
              </div>
              <div className="flex justify-center text-slate-300 dark:text-slate-700">↓</div>
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-700 text-xs font-bold text-white dark:bg-slate-800">
                  5
                </span>
                <div className="flex w-full items-center justify-between rounded-lg bg-gradient-to-r from-slate-100 to-slate-200 px-3 py-2 text-xs font-bold text-slate-800 border border-slate-300 dark:from-slate-800 dark:to-slate-700 dark:border-slate-600 dark:text-slate-200">
                  <span>POSITIVE (1) or NEGATIVE (0)</span>
                  <span className="text-[11px] font-normal text-slate-600 dark:text-slate-400">
                    + Decision Margin
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
