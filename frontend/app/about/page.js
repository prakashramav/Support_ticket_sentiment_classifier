import Link from "next/link";
import {
  Info,
  ShieldCheck,
  Film,
  Database,
  Layers,
  ArrowLeft,
  ChevronRight,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Code2,
} from "lucide-react";

export default function AboutPage() {
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
            About Project & Limitations
          </span>
        </div>

        {/* Page Header */}
        <div className="border-b border-slate-200/80 pb-6 dark:border-slate-800">
          <div className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300">
            <Info className="h-3.5 w-3.5 text-indigo-600" />
            <span>Project Background & Scope</span>
          </div>
          <h1 className="mt-2 text-2xl font-extrabold text-slate-900 sm:text-3xl dark:text-white">
            About Support-Ticket Sentiment Classifier
          </h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300 max-w-3xl">
            An academic, educational decision-support web application designed to demonstrate
            the differences between state-of-the-art Transformer models (BERT) and classical NLP baselines (TF-IDF + Linear SVM).
          </p>
        </div>

        {/* Section 1: The Core Objective */}
        <div className="mt-10 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="rounded-3xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-900">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 dark:bg-indigo-950 dark:text-indigo-400">
              <Sparkles className="h-5 w-5" />
            </div>
            <h3 className="mt-4 text-base font-bold text-slate-900 dark:text-white">
              Decision Support
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
              Designed as an assistive triage aid for support representatives, helping prioritize high-frustration tickets rather than replacing human staff.
            </p>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-900">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-950 dark:text-purple-400">
              <Layers className="h-5 w-5" />
            </div>
            <h3 className="mt-4 text-base font-bold text-slate-900 dark:text-white">
              Comparative Analysis
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
              Evaluates whether the computational cost of a 110M-parameter Transformer translates into meaningful sentiment gains over lightweight linear models.
            </p>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-900">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-950 dark:text-emerald-400">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <h3 className="mt-4 text-base font-bold text-slate-900 dark:text-white">
              Transparent Engineering
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
              Openly documents the SST-2 benchmark dataset origins, evaluation splits, and the real-world domain gap without compromising scientific validity.
            </p>
          </div>
        </div>

        {/* Section 2: Deep Dive into Domain Gap */}
        <div id="domain-gap" className="mt-12 rounded-3xl border border-amber-200 bg-amber-50/50 p-6 sm:p-8 dark:border-amber-900/50 dark:bg-amber-950/20">
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-500 text-white">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-amber-950 dark:text-amber-200">
                Understanding the SST-2 Dataset Domain Gap
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-amber-900/90 dark:text-amber-300/90">
                The Stanford Sentiment Treebank (SST-2), part of the GLUE benchmark, is composed of
                movie-review snippets from Rotten Tomatoes. While it serves as the premier academic standard
                for evaluating binary sentiment models, customer support tickets present distinct linguistic characteristics:
              </p>

              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div className="rounded-2xl border border-amber-200/80 bg-white/90 p-4 dark:border-amber-900/60 dark:bg-slate-900">
                  <span className="font-bold text-slate-900 dark:text-white block mb-1">
                    SST-2 Movie Review Sentences
                  </span>
                  <ul className="space-y-1 text-slate-600 dark:text-slate-400">
                    <li>• Eloquent adjectives, artistic descriptions, film critique syntax.</li>
                    <li>• Generally grammatically complete single sentences.</li>
                    <li>• Neutral or sarcastic humor common in theatrical writing.</li>
                  </ul>
                </div>

                <div className="rounded-2xl border border-amber-200/80 bg-white/90 p-4 dark:border-amber-900/60 dark:bg-slate-900">
                  <span className="font-bold text-slate-900 dark:text-white block mb-1">
                    Real Enterprise Support Tickets
                  </span>
                  <ul className="space-y-1 text-slate-600 dark:text-slate-400">
                    <li>• Domain jargon, billing references, account IDs, product SKU codes.</li>
                    <li>• Urgent emotional register, truncated sentences, and punctuation emphasis.</li>
                    <li>• Multi-turn conversational history and follow-up clarifications.</li>
                  </ul>
                </div>
              </div>

              <p className="mt-4 text-xs text-amber-900/80 dark:text-amber-300/80">
                <strong>Viva Takeaway:</strong> Communicating this domain boundary demonstrates mature machine learning engineering. The model serves as an effective proof-of-concept; production readiness requires fine-tuning on domain-specific ticket corpora.
              </p>
            </div>
          </div>
        </div>

        {/* Section 3: Architecture Breakdown */}
        <div className="mt-12 rounded-3xl border border-slate-200 bg-white p-6 sm:p-8 dark:border-slate-800 dark:bg-slate-900">
          <div className="flex items-center gap-3 border-b border-slate-100 pb-4 dark:border-slate-800">
            <Code2 className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">
              End-to-End System Architecture
            </h3>
          </div>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-slate-600 dark:text-slate-400">
            <div>
              <span className="font-bold text-slate-900 dark:text-white text-sm block mb-1">
                1. Frontend Client
              </span>
              <p>Next.js 16 with React 19 and Tailwind CSS. Provides an intuitive SaaS layout, character counter, live predictions, and responsive design for mobile, tablet, and desktop.</p>
            </div>
            <div>
              <span className="font-bold text-slate-900 dark:text-white text-sm block mb-1">
                2. API Service Layer
              </span>
              <p>FastAPI async Python server with Pydantic request validation and CORS support, communicating via Axios with client-side fallback robustness.</p>
            </div>
            <div>
              <span className="font-bold text-slate-900 dark:text-white text-sm block mb-1">
                3. Model Layer
              </span>
              <p>Hugging Face Transformers (BERT-base-uncased) and scikit-learn (TF-IDF + LinearSVC) with standardized SST-2 GLUE benchmark validation.</p>
            </div>
          </div>

          <div className="mt-8 flex justify-center border-t border-slate-100 pt-6 dark:border-slate-800">
            <Link
              href="/analyze"
              className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-6 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500"
            >
              <span>Launch Live Analyzer Studio</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
