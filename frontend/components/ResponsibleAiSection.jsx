import {
  ShieldCheck,
  AlertTriangle,
  Database,
  Film,
  CheckCircle2,
  Lock,
  Eye,
} from "lucide-react";
import Link from "next/link";

export default function ResponsibleAiSection() {
  const cards = [
    {
      title: "Dataset",
      value: "SST-2 / GLUE",
      detail: "Stanford Sentiment Treebank (v2) from General Language Understanding Evaluation benchmark.",
      icon: Database,
    },
    {
      title: "Domain",
      value: "Movie-review sentences",
      detail: "Single-sentence snippets extracted from Rotten Tomatoes film critiques.",
      icon: Film,
    },
    {
      title: "Classification",
      value: "Positive / Negative",
      detail: "Binary sentiment polarities mapped to 1 (positive sentiment) or 0 (negative sentiment).",
      icon: CheckCircle2,
    },
  ];

  return (
    <section className="py-20 md:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="mx-auto max-w-3xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 mb-3 border border-emerald-200 dark:border-emerald-800">
            <ShieldCheck className="h-3.5 w-3.5" />
            <span>Ethical AI & Scope Disclosures</span>
          </div>
          <h2 className="text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
            Built With Transparency
          </h2>
          <p className="mt-4 text-base text-slate-600 dark:text-slate-300">
            Machine learning systems in customer service must be transparent about
            their origins, data distributions, and intended operational boundaries.
          </p>
        </div>

        {/* 3 Cards */}
        <div className="mt-14 grid grid-cols-1 gap-6 sm:grid-cols-3">
          {cards.map((item) => {
            const Icon = item.icon;
            return (
              <div
                key={item.title}
                className="flex flex-col justify-between rounded-3xl border border-slate-200/80 bg-white p-7 shadow-xs dark:border-slate-800 dark:bg-slate-900"
              >
                <div>
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 dark:bg-indigo-950/60 dark:text-indigo-400">
                    <Icon className="h-5 w-5" />
                  </div>
                  <span className="mt-5 block text-xs font-bold uppercase tracking-wider text-slate-400">
                    {item.title}
                  </span>
                  <h3 className="mt-1 text-xl font-extrabold text-slate-900 dark:text-white">
                    {item.value}
                  </h3>
                  <p className="mt-3 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                    {item.detail}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Highlighted Disclaimer Callout */}
        <div className="mt-10 rounded-3xl border border-amber-300/80 bg-gradient-to-br from-amber-50/90 via-orange-50/50 to-amber-50/80 p-6 sm:p-8 shadow-sm dark:border-amber-900/60 dark:from-amber-950/30 dark:via-slate-900 dark:to-amber-950/20">
          <div className="flex flex-col sm:flex-row items-start gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-500 text-white shadow-md shadow-amber-500/20">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h4 className="text-base font-bold text-amber-950 dark:text-amber-200">
                Important Domain Disclosure & Prototype Boundaries
              </h4>
              <p className="mt-2 text-sm leading-relaxed text-amber-900/90 dark:text-amber-300/90">
                <strong>Important:</strong> SST-2 contains movie-review text rather
                than real customer-support tickets. This project is an educational
                decision-support prototype. Real support-ticket deployment would require
                domain-specific training data, validation and human oversight.
              </p>
              <div className="mt-4 flex flex-wrap items-center gap-4 text-xs font-semibold text-amber-900 dark:text-amber-200">
                <Link
                  href="/about#domain-gap"
                  className="inline-flex items-center gap-1 underline underline-offset-4 hover:text-amber-700 dark:hover:text-amber-100"
                >
                  <span>Read our full Domain Gap Analysis</span>
                  <span>→</span>
                </Link>
                <span>•</span>
                <Link
                  href="/model"
                  className="inline-flex items-center gap-1 underline underline-offset-4 hover:text-amber-700 dark:hover:text-amber-100"
                >
                  <span>Examine Model Card Intended Use</span>
                  <span>→</span>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
