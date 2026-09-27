import Link from "next/link";
import { Cpu, ExternalLink, ShieldCheck, Sparkles, Code2 } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-slate-200/80 bg-slate-50/70 dark:border-slate-800/80 dark:bg-slate-950">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-5">
          {/* Col 1: Brand & Overview */}
          <div className="lg:col-span-2 space-y-4">
            <Link href="/" className="inline-flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-sm">
                <Cpu className="h-5 w-5" />
              </div>
              <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">
                Sentiment<span className="text-indigo-600 dark:text-indigo-400">AI</span>
              </span>
            </Link>
            <p className="max-w-md text-sm leading-relaxed text-slate-600 dark:text-slate-400">
              An educational NLP sentiment-classification project using BERT and a
              TF-IDF baseline. Designed to explore contextual Transformer representations
              versus classical frequency features on binary text classification.
            </p>
            <div className="flex items-center gap-3 pt-2">
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-2xs hover:bg-slate-50 hover:text-indigo-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-indigo-400"
              >
                <svg className="h-3.5 w-3.5 fill-current" viewBox="0 0 24 24">
                  <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
                </svg>
                <span>GitHub Repository</span>
                <ExternalLink className="h-3 w-3 opacity-60" />
              </a>
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-700 ring-1 ring-emerald-600/20 dark:bg-emerald-950/50 dark:text-emerald-300">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                SST-2 (GLUE) Prototype
              </span>
            </div>
          </div>

          {/* Col 2: Product */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-200">
              Product
            </h3>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li>
                <Link
                  href="/analyze"
                  className="text-slate-600 transition-colors hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400"
                >
                  Analyze Text
                </Link>
              </li>
              <li>
                <Link
                  href="/performance"
                  className="text-slate-600 transition-colors hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400"
                >
                  Performance Dashboard
                </Link>
              </li>
              <li>
                <Link
                  href="/model"
                  className="text-slate-600 transition-colors hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400"
                >
                  Model Architecture
                </Link>
              </li>
            </ul>
          </div>

          {/* Col 3: Resources */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-200">
              Resources
            </h3>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li>
                <Link
                  href="/about"
                  className="text-slate-600 transition-colors hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400"
                >
                  About Project
                </Link>
              </li>
              <li>
                <Link
                  href="/model#card"
                  className="text-slate-600 transition-colors hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400"
                >
                  Model Card & Limitations
                </Link>
              </li>
              <li>
                <Link
                  href="/about#domain-gap"
                  className="text-slate-600 transition-colors hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400"
                >
                  Dataset Domain Gap
                </Link>
              </li>
            </ul>
          </div>

          {/* Col 4: Technology */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-200">
              Technology
            </h3>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li className="text-slate-600 dark:text-slate-400">BERT (Hugging Face)</li>
              <li className="text-slate-600 dark:text-slate-400">TF-IDF + Linear SVM</li>
              <li className="text-slate-600 dark:text-slate-400">FastAPI & Pydantic</li>
              <li className="text-slate-600 dark:text-slate-400">Next.js 16 (App Router)</li>
            </ul>
          </div>
        </div>

        {/* Bottom Disclaimer Banner */}
        <div className="mt-12 rounded-2xl border border-amber-200/70 bg-amber-50/60 p-4 dark:border-amber-900/40 dark:bg-amber-950/20">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between text-xs text-amber-900 dark:text-amber-200">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
              <span className="font-semibold">Responsible AI Statement:</span>
              <span>
                Educational decision-support prototype. Not intended for autonomous customer-facing decisions.
              </span>
            </div>
            <span className="text-amber-700/80 dark:text-amber-300/80 text-[11px]">
              SST-2 Benchmark Movie Review Domain
            </span>
          </div>
        </div>

        {/* Copyright */}
        <div className="mt-8 flex flex-col items-center justify-between gap-4 border-t border-slate-200/60 pt-6 text-xs text-slate-500 sm:flex-row dark:border-slate-800/60 dark:text-slate-400">
          <p>© {new Date().getFullYear()} Support-Ticket Sentiment Classifier. Academic & Research Prototype.</p>
          <div className="flex items-center gap-4">
            <Link href="/about" className="hover:underline">Transparency</Link>
            <Link href="/model" className="hover:underline">Model Card</Link>
            <Link href="/performance" className="hover:underline">Benchmarks</Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
