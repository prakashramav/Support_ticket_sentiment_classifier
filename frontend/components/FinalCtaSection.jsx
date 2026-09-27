import Link from "next/link";
import { ArrowRight, ArrowUpRight, Sparkles, Cpu } from "lucide-react";

export default function FinalCtaSection() {
  return (
    <section className="py-20 md:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="relative overflow-hidden rounded-3xl border border-indigo-200/80 bg-gradient-to-tr from-indigo-900 via-indigo-800 to-slate-950 p-8 sm:p-12 lg:p-16 text-center text-white shadow-2xl shadow-indigo-950/20 dark:border-indigo-800/80">
          {/* Subtle glowing ambient orbs */}
          <div
            className="pointer-events-none absolute -top-24 -left-24 h-96 w-96 rounded-full bg-indigo-500/20 blur-3xl"
            aria-hidden="true"
          />
          <div
            className="pointer-events-none absolute -bottom-24 -right-24 h-96 w-96 rounded-full bg-violet-500/20 blur-3xl"
            aria-hidden="true"
          />

          <div className="relative mx-auto max-w-3xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-indigo-400/30 bg-indigo-500/10 px-3.5 py-1 text-xs font-semibold text-indigo-200 backdrop-blur-xs">
              <Sparkles className="h-3.5 w-3.5 text-indigo-300" />
              <span>Fast & Explainable NLP Prototype</span>
            </div>

            <h2 className="mt-6 text-3xl font-extrabold tracking-tight sm:text-4xl lg:text-5xl">
              Ready to Analyze Your Text?
            </h2>

            <p className="mt-4 text-base sm:text-lg text-indigo-100/80 leading-relaxed">
              Turn text into an actionable sentiment signal with BERT.
            </p>

            <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/analyze"
                className="group w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-white px-6 py-3.5 text-base font-bold text-indigo-950 shadow-md transition-all hover:bg-indigo-50 hover:shadow-lg active:scale-[0.98]"
              >
                <span>Start Analyzing</span>
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Link>
              <Link
                href="/performance"
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl border border-indigo-400/40 bg-indigo-950/40 px-6 py-3.5 text-base font-semibold text-white backdrop-blur-xs transition-all hover:bg-indigo-900/60"
              >
                <span>View Model Performance</span>
                <ArrowUpRight className="h-4 w-4 opacity-70" />
              </Link>
            </div>

            <p className="mt-6 text-xs text-indigo-200/60">
              Free educational & viva demonstration prototype • No authentication required
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
