import {
  Cpu,
  Binary,
  Percent,
  BarChart3,
  Info,
  ShieldCheck,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";

export default function FeatureSection() {
  const features = [
    {
      title: "BERT Sentiment Analysis",
      description:
        "Fine-tuned pretrained BERT model for binary sentiment classification.",
      icon: Cpu,
      href: "/analyze",
      color: "text-indigo-600 dark:text-indigo-400",
      bg: "bg-indigo-50 dark:bg-indigo-950/60",
    },
    {
      title: "TF-IDF Baseline",
      description:
        "Traditional TF-IDF + Linear SVM pipeline for model comparison.",
      icon: Binary,
      href: "/analyze",
      color: "text-purple-600 dark:text-purple-400",
      bg: "bg-purple-50 dark:bg-purple-950/60",
    },
    {
      title: "Confidence Score",
      description:
        "See how strongly the model supports its prediction.",
      icon: Percent,
      href: "/analyze",
      color: "text-emerald-600 dark:text-emerald-400",
      bg: "bg-emerald-50 dark:bg-emerald-950/60",
    },
    {
      title: "Model Comparison",
      description:
        "Compare BERT and the baseline using Accuracy and F1-score.",
      icon: BarChart3,
      href: "/performance",
      color: "text-blue-600 dark:text-blue-400",
      bg: "bg-blue-50 dark:bg-blue-950/60",
    },
    {
      title: "Model Information",
      description:
        "View model version, dataset information and evaluation details.",
      icon: Info,
      href: "/model",
      color: "text-teal-600 dark:text-teal-400",
      bg: "bg-teal-50 dark:bg-teal-950/60",
    },
    {
      title: "Responsible AI",
      description:
        "Understand intended use, limitations and the dataset domain gap.",
      icon: ShieldCheck,
      href: "/about",
      color: "text-amber-600 dark:text-amber-400",
      bg: "bg-amber-50 dark:bg-amber-950/60",
    },
  ];

  return (
    <section className="py-20 md:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mx-auto max-w-3xl text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-indigo-600 dark:text-indigo-400">
            System Capabilities
          </span>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
            Everything You Need to Analyze Sentiment
          </h2>
          <p className="mt-4 text-base text-slate-600 dark:text-slate-300">
            Built as an accessible, rigorous machine learning environment for evaluating
            contextual versus lexical sentiment classifiers.
          </p>
        </div>

        {/* 6 Cards Grid */}
        <div className="mt-14 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.title}
                href={item.href}
                className="group relative flex flex-col justify-between rounded-3xl border border-slate-200/80 bg-white p-7 shadow-2xs transition-all duration-200 hover:-translate-y-1 hover:border-indigo-300 hover:shadow-lg dark:border-slate-800 dark:bg-slate-900 dark:hover:border-indigo-700"
              >
                <div>
                  <div
                    className={`flex h-12 w-12 items-center justify-center rounded-2xl ${item.bg} ${item.color} transition-transform group-hover:scale-110`}
                  >
                    <Icon className="h-6 w-6" />
                  </div>
                  <h3 className="mt-5 text-lg font-bold text-slate-900 dark:text-white">
                    {item.title}
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
                    {item.description}
                  </p>
                </div>

                <div className="mt-6 flex items-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 transition-colors group-hover:text-indigo-500">
                  <span>Explore detail</span>
                  <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-1" />
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </section>
  );
}
