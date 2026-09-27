import { Cpu, Layers, Database, Zap, Binary, Box, Workflow } from "lucide-react";

export default function TechStrip() {
  const technologies = [
    {
      name: "BERT",
      category: "Transformer",
      description: "bert-base-uncased",
      icon: Cpu,
      color: "from-blue-500 to-indigo-600",
    },
    {
      name: "Hugging Face",
      category: "Ecosystem",
      description: "Transformers & Datasets",
      icon: Box,
      color: "from-amber-500 to-orange-600",
    },
    {
      name: "SST-2 / GLUE",
      category: "Benchmark",
      description: "Stanford Sentiment Treebank",
      icon: Database,
      color: "from-emerald-500 to-teal-600",
    },
    {
      name: "TF-IDF",
      category: "Feature Extractor",
      description: "N-gram Vectorizer",
      icon: Binary,
      color: "from-purple-500 to-pink-600",
    },
    {
      name: "Linear SVM",
      category: "Baseline Model",
      description: "scikit-learn Support Vector",
      icon: Layers,
      color: "from-rose-500 to-red-600",
    },
    {
      name: "FastAPI",
      category: "Backend Engine",
      description: "Asynchronous Python 3.11",
      icon: Zap,
      color: "from-teal-500 to-cyan-600",
    },
    {
      name: "Next.js",
      category: "Frontend App",
      description: "React 19 & Tailwind CSS",
      icon: Workflow,
      color: "from-slate-700 to-slate-900 dark:from-slate-300 dark:to-slate-100",
    },
  ];

  return (
    <section className="border-y border-slate-200/80 bg-slate-50/50 py-10 dark:border-slate-800/80 dark:bg-slate-950/40">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-6">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 dark:text-slate-400">
            Built as an end-to-end NLP and machine-learning demonstration.
          </p>
        </div>

        {/* Technology Badges Grid */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7">
          {technologies.map((tech) => {
            const Icon = tech.icon;
            return (
              <div
                key={tech.name}
                className="group relative flex flex-col items-center justify-center rounded-2xl border border-slate-200/70 bg-white/80 p-3.5 text-center shadow-2xs transition-all duration-200 hover:-translate-y-0.5 hover:border-indigo-300 hover:shadow-md dark:border-slate-800 dark:bg-slate-900/80 dark:hover:border-indigo-700"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 transition-colors group-hover:bg-indigo-50 group-hover:text-indigo-600 dark:group-hover:bg-indigo-950/60 dark:group-hover:text-indigo-400">
                  <Icon className="h-4 w-4" />
                </div>
                <span className="mt-2 text-xs font-bold text-slate-800 dark:text-slate-100">
                  {tech.name}
                </span>
                <span className="text-[10px] text-slate-400 dark:text-slate-500">
                  {tech.category}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
