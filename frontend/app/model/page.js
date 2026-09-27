import Link from "next/link";
import {
  BookOpen,
  Cpu,
  Database,
  Binary,
  ShieldCheck,
  ArrowLeft,
  ChevronRight,
  Layers,
  Sparkles,
  Sliders,
  CheckCircle2,
  FileCode,
} from "lucide-react";

export default function ModelPage() {
  const modelSpecs = [
    { label: "Base Architecture", value: "BERT (Bidirectional Encoder Representations from Transformers)" },
    { label: "Checkpoint Base", value: "bert-base-uncased (Hugging Face)" },
    { label: "Model Parameters", value: "110 Million Parameters (12 layers, 768 hidden, 12 attention heads)" },
    { label: "Vocabulary Size", value: "30,522 tokens (WordPiece tokenizer)" },
    { label: "Max Sequence Length", value: "128 tokens" },
    { label: "Task Definition", value: "Binary Sequence Classification" },
    { label: "Target Labels", value: "0 = Negative, 1 = Positive" },
    { label: "Dataset Benchmark", value: "SST-2 (Stanford Sentiment Treebank v2 - GLUE)" },
    { label: "Dataset Size", value: "67,349 training sentences, 872 validation sentences" },
  ];

  const baselineSpecs = [
    { label: "Baseline Model", value: "Linear Support Vector Machine (LinearSVC via scikit-learn)" },
    { label: "Feature Extraction", value: "TF-IDF Vectorizer (Term Frequency × Inverse Document Frequency)" },
    { label: "N-gram Range", value: "(1, 2) Unigrams & Bigrams" },
    { label: "Max Features", value: "10,000 vocabulary features" },
    { label: "Hardware Footprint", value: "< 20MB memory, zero GPU requirement" },
  ];

  return (
    <div className="py-10 md:py-16">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Breadcrumb */}
        <div className="mb-6 flex items-center gap-2 text-xs font-medium text-slate-500">
          <Link href="/" className="hover:text-indigo-600 flex items-center gap-1">
            <ArrowLeft className="h-3.5 w-3.5" />
            <span>Back to Landing</span>
          </Link>
          <ChevronRight className="h-3.5 w-3.5" />
          <span className="text-slate-900 dark:text-white font-semibold">
            Model Specifications & Card
          </span>
        </div>

        {/* Page Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-200/80 pb-6 dark:border-slate-800">
          <div>
            <div className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300">
              <BookOpen className="h-3.5 w-3.5 text-indigo-600" />
              <span>Hugging Face Style Model Card</span>
            </div>
            <h1 className="mt-2 text-2xl font-extrabold text-slate-900 sm:text-3xl dark:text-white">
              Model Information & Governance
            </h1>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
              Full architectural details, vocabulary size, hyperparameters, and intended operational scope.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/analyze"
              className="inline-flex items-center gap-1.5 rounded-xl bg-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow-2xs hover:bg-indigo-500"
            >
              <Sparkles className="h-3.5 w-3.5" />
              <span>Test Model in Studio</span>
            </Link>
          </div>
        </div>

        {/* Primary Spec Grid */}
        <div className="mt-10 grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* BERT Card */}
          <div className="rounded-3xl border border-indigo-200/90 bg-white p-6 sm:p-8 shadow-sm dark:border-indigo-900/60 dark:bg-slate-900">
            <div className="flex items-center gap-3 border-b border-slate-100 pb-4 dark:border-slate-800">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-white">
                <Cpu className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-slate-900 dark:text-white">
                  BERT Sentiment Classifier
                </h2>
                <span className="text-xs text-indigo-600 dark:text-indigo-400 font-mono">
                  Primary Transformer Pipeline
                </span>
              </div>
            </div>

            <div className="mt-6 divide-y divide-slate-100 dark:divide-slate-800">
              {modelSpecs.map((item) => (
                <div
                  key={item.label}
                  className="flex flex-col sm:flex-row sm:items-center sm:justify-between py-3 text-xs sm:text-sm"
                >
                  <span className="font-semibold text-slate-500 dark:text-slate-400">
                    {item.label}
                  </span>
                  <span className="mt-0.5 sm:mt-0 font-medium text-slate-900 dark:text-slate-200 text-right">
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Baseline Spec Card */}
          <div className="rounded-3xl border border-slate-200 bg-white p-6 sm:p-8 shadow-sm dark:border-slate-800 dark:bg-slate-900">
            <div className="flex items-center gap-3 border-b border-slate-100 pb-4 dark:border-slate-800">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-700 text-white dark:bg-slate-800">
                <Binary className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-slate-900 dark:text-white">
                  TF-IDF + Linear SVM Baseline
                </h2>
                <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                  Classical Benchmark Pipeline
                </span>
              </div>
            </div>

            <div className="mt-6 divide-y divide-slate-100 dark:divide-slate-800">
              {baselineSpecs.map((item) => (
                <div
                  key={item.label}
                  className="flex flex-col sm:flex-row sm:items-center sm:justify-between py-3 text-xs sm:text-sm"
                >
                  <span className="font-semibold text-slate-500 dark:text-slate-400">
                    {item.label}
                  </span>
                  <span className="mt-0.5 sm:mt-0 font-medium text-slate-900 dark:text-slate-200 text-right">
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Intended Use & Limitations Section */}
        <div id="card" className="mt-10 rounded-3xl border border-slate-200 bg-slate-50/70 p-6 sm:p-8 dark:border-slate-800 dark:bg-slate-950/60">
          <div className="flex items-center gap-3">
            <ShieldCheck className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">
              Intended Use & Operational Limitations
            </h3>
          </div>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-600 dark:text-slate-300">
            <div className="rounded-2xl bg-white p-5 border border-slate-200 dark:bg-slate-900 dark:border-slate-800">
              <h4 className="font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                <span>Intended Application</span>
              </h4>
              <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                Educational demonstrations, NLP coursework viva, research benchmarking, and decision-support assistance for human customer service agents prioritizing response queues.
              </p>
            </div>

            <div className="rounded-2xl bg-white p-5 border border-slate-200 dark:bg-slate-900 dark:border-slate-800">
              <h4 className="font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-rose-500" />
                <span>Out-of-Scope Use Cases</span>
              </h4>
              <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                Autonomous customer penalty decisions, automated account bans, unmonitored ticket resolution, or critical escalation without human verification.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
