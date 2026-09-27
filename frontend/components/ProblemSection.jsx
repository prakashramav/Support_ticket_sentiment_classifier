import { Inbox, Clock, Scale, ArrowRight, ShieldCheck } from "lucide-react";
import Link from "next/link";

export default function ProblemSection() {
  const problems = [
    {
      title: "High Volume",
      subtitle: "Support teams deal with large amounts of text.",
      description:
        "Modern customer support channels receive thousands of inbound tickets, chats, and emails every single day, easily overwhelming triage workflows.",
      icon: Inbox,
      badge: "Scale Challenge",
    },
    {
      title: "Manual Analysis",
      subtitle: "Reading every message manually takes time.",
      description:
        "Triage representatives spend critical hours reading raw text line-by-line just to identify frustrated customers requiring urgent escalation.",
      icon: Clock,
      badge: "Latency Bottleneck",
    },
    {
      title: "Inconsistent Interpretation",
      subtitle: "Different people may interpret the same message differently.",
      description:
        "Human emotion perception varies by agent workload, shift fatigue, and subjective bias, leading to unpredictable priority tagging.",
      icon: Scale,
      badge: "Variance & Subjectivity",
    },
  ];

  return (
    <section className="py-20 md:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="mx-auto max-w-3xl text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-indigo-600 dark:text-indigo-400">
            The Operational Challenge
          </span>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
            Turn Unstructured Text Into a Clear Sentiment Signal
          </h2>
          <p className="mt-4 text-base text-slate-600 dark:text-slate-300">
            Unstructured customer feedback contains vital satisfaction signals,
            yet traditional manual review struggles to keep pace with modern message velocity.
          </p>
        </div>

        {/* 3 Problem Cards */}
        <div className="mt-14 grid grid-cols-1 gap-8 md:grid-cols-3">
          {problems.map((item, idx) => {
            const Icon = item.icon;
            return (
              <div
                key={item.title}
                className="group relative flex flex-col justify-between rounded-3xl border border-slate-200/80 bg-white p-8 shadow-xs transition-all duration-300 hover:-translate-y-1 hover:border-indigo-300 hover:shadow-xl dark:border-slate-800 dark:bg-slate-900 dark:hover:border-indigo-700"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-50 text-indigo-600 dark:bg-indigo-950/60 dark:text-indigo-400">
                      <Icon className="h-6 w-6" />
                    </div>
                    <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-semibold text-slate-600 dark:bg-slate-800 dark:text-slate-400">
                      0{idx + 1}
                    </span>
                  </div>

                  <h3 className="mt-6 text-xl font-bold text-slate-900 dark:text-white">
                    {item.title}
                  </h3>
                  <p className="mt-2 text-sm font-semibold text-indigo-600 dark:text-indigo-400">
                    {item.subtitle}
                  </p>
                  <p className="mt-3 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
                    {item.description}
                  </p>
                </div>

                <div className="mt-6 border-t border-slate-100 pt-4 dark:border-slate-800">
                  <span className="text-xs font-medium text-slate-400">
                    Impact: {item.badge}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Key Resolution Statement Callout */}
        <div className="mt-12 overflow-hidden rounded-3xl border border-indigo-200/90 bg-gradient-to-r from-indigo-50/90 via-violet-50/70 to-sky-50/80 p-8 shadow-sm dark:border-indigo-900/60 dark:from-indigo-950/40 dark:via-purple-950/30 dark:to-slate-900">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6 text-center md:text-left">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-indigo-600 text-white shadow-md shadow-indigo-600/30">
                <ShieldCheck className="h-6 w-6" />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
                  Decision Support Solution
                </p>
                <p className="text-lg font-bold text-slate-900 dark:text-white sm:text-xl">
                  “Our system provides a fast, consistent sentiment signal to
                  support human decision-making.”
                </p>
              </div>
            </div>
            <Link
              href="/analyze"
              className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition-all hover:bg-slate-800 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-100"
            >
              <span>See It In Action</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
