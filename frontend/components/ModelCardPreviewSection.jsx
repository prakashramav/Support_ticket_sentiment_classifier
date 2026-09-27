import Link from "next/link";
import { Cpu, ArrowRight, BookOpen, Layers, CheckCircle2 } from "lucide-react";

export default function ModelCardPreviewSection() {
  const specs = [
    { label: "Model", value: "BERT Sentiment Classifier" },
    { label: "Dataset", value: "SST-2 / GLUE" },
    { label: "Task", value: "Binary Sentiment Classification" },
    { label: "Labels", value: "Positive / Negative" },
    { label: "Architecture", value: "Pretrained BERT + Classification Head" },
  ];

  return (
    <section className="border-t border-slate-200/80 bg-slate-50/50 py-20 dark:border-slate-800/80 dark:bg-slate-950/50 md:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-indigo-600 dark:text-indigo-400">
            Model Governance & Specifications
          </span>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl dark:text-white">
            Know What the Model Knows
          </h2>
          <p className="mt-4 text-base text-slate-600 dark:text-slate-300">
            Transparent documentation adhering to modern Hugging Face model card standards.
          </p>
        </div>

        {/* Model Card Box */}
        <div className="mt-12 mx-auto max-w-3xl rounded-3xl border border-slate-200/90 bg-white p-6 sm:p-8 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <div className="flex items-center justify-between border-b border-slate-100 pb-4 dark:border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-white">
                <BookOpen className="h-4 w-4" />
              </div>
              <div>
                <h3 className="text-base font-bold text-slate-900 dark:text-white">
                  Model Specification Card
                </h3>
                <span className="text-xs text-slate-400">
                  Standardized Hugging Face Architecture Spec
                </span>
              </div>
            </div>
            <span className="rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-mono font-semibold text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
              v1.0.0
            </span>
          </div>

          {/* Key Spec Rows */}
          <div className="mt-6 divide-y divide-slate-100 dark:divide-slate-800">
            {specs.map((item) => (
              <div
                key={item.label}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between py-3.5 text-sm"
              >
                <span className="font-semibold text-slate-500 dark:text-slate-400">
                  {item.label}
                </span>
                <span className="mt-1 sm:mt-0 font-medium text-slate-900 dark:text-white">
                  {item.value}
                </span>
              </div>
            ))}
          </div>

          {/* Button CTA */}
          <div className="mt-8 flex justify-center border-t border-slate-100 pt-6 dark:border-slate-800">
            <Link
              href="/model"
              className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-6 py-3 text-sm font-semibold text-white shadow-sm transition-all hover:bg-indigo-500 active:scale-[0.98]"
            >
              <span>View Model Card</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
