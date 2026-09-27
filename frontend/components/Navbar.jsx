"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Cpu,
  Menu,
  X,
  Sparkles,
  ArrowRight,
  BarChart3,
  BookOpen,
  Info,
  Terminal,
} from "lucide-react";

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();

  const navLinks = [
    { name: "Home", href: "/" },
    { name: "Analyze", href: "/analyze" },
    { name: "Performance", href: "/performance" },
    { name: "Model", href: "/model" },
    { name: "About", href: "/about" },
  ];

  const isActive = (href) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-200/80 bg-white/85 backdrop-blur-md transition-colors dark:border-slate-800/80 dark:bg-slate-950/85">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand Logo */}
        <Link
          href="/"
          className="group flex items-center gap-2.5 transition-transform active:scale-95"
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-violet-500 text-white shadow-sm shadow-indigo-500/25 ring-1 ring-white/20">
            <Cpu className="h-5 w-5 transition-transform duration-300 group-hover:rotate-12" />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <span className="text-lg font-bold tracking-tight text-slate-900 dark:text-white">
                Sentiment<span className="text-indigo-600 dark:text-indigo-400">AI</span>
              </span>
              <span className="rounded-full bg-indigo-50 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-700 ring-1 ring-indigo-600/10 dark:bg-indigo-950/60 dark:text-indigo-300 dark:ring-indigo-400/20">
                BERT
              </span>
            </div>
            <span className="text-[10px] font-medium text-slate-500 dark:text-slate-400 -mt-1 hidden sm:block">
              Support-Ticket Classifier
            </span>
          </div>
        </Link>

        {/* Desktop Navigation Links */}
        <nav className="hidden items-center gap-1 md:flex">
          {navLinks.map((link) => (
            <Link
              key={link.name}
              href={link.href}
              className={`relative rounded-lg px-3.5 py-2 text-sm font-medium transition-all ${
                isActive(link.href)
                  ? "text-indigo-600 dark:text-indigo-400 font-semibold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-100/60 dark:text-slate-300 dark:hover:text-white dark:hover:bg-slate-900/60"
              }`}
            >
              {link.name}
              {isActive(link.href) && (
                <span className="absolute inset-x-3 bottom-0.5 h-0.5 rounded-full bg-indigo-600 dark:bg-indigo-400" />
              )}
            </Link>
          ))}
        </nav>

        {/* Right CTA */}
        <div className="hidden items-center gap-3 md:flex">
          <Link
            href="/analyze"
            className="group relative inline-flex items-center gap-2 overflow-hidden rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-4 py-2 text-sm font-semibold text-white shadow-sm shadow-indigo-600/20 transition-all hover:shadow-indigo-600/40 hover:from-indigo-500 hover:to-violet-500 active:scale-[0.98]"
          >
            <span>Try Analyzer</span>
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
        </div>

        {/* Mobile menu toggle */}
        <button
          type="button"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="inline-flex items-center justify-center rounded-lg p-2 text-slate-600 hover:bg-slate-100 hover:text-slate-900 focus:outline-none dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white md:hidden"
          aria-label="Toggle Navigation Menu"
        >
          {mobileMenuOpen ? (
            <X className="h-6 w-6" />
          ) : (
            <Menu className="h-6 w-6" />
          )}
        </button>
      </div>

      {/* Mobile dropdown */}
      {mobileMenuOpen && (
        <div className="border-b border-slate-200/80 bg-white/95 px-4 pt-2 pb-5 shadow-lg backdrop-blur-md dark:border-slate-800/80 dark:bg-slate-950/95 md:hidden">
          <div className="flex flex-col space-y-1">
            {navLinks.map((link) => (
              <Link
                key={link.name}
                href={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`rounded-lg px-3 py-2.5 text-base font-medium transition-colors ${
                  isActive(link.href)
                    ? "bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300"
                    : "text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-900"
                }`}
              >
                {link.name}
              </Link>
            ))}
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800">
            <Link
              href="/analyze"
              onClick={() => setMobileMenuOpen(false)}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:bg-indigo-500"
            >
              <span>Try Analyzer</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
