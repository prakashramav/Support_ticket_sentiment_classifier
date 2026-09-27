import HeroSection from "../components/HeroSection";
import TechStrip from "../components/TechStrip";
import ProblemSection from "../components/ProblemSection";
import HowItWorksSection from "../components/HowItWorksSection";
import FeatureSection from "../components/FeatureSection";
import LiveAnalyzerSection from "../components/LiveAnalyzerSection";
import ComparisonSection from "../components/ComparisonSection";
import PerformancePreviewSection from "../components/PerformancePreviewSection";
import ResponsibleAiSection from "../components/ResponsibleAiSection";
import ModelCardPreviewSection from "../components/ModelCardPreviewSection";
import FinalCtaSection from "../components/FinalCtaSection";

export default function Home() {
  return (
    <div className="relative">
      {/* 2. Hero Section */}
      <HeroSection />

      {/* 3. Trust / Technology Strip */}
      <TechStrip />

      {/* 4. Problem Section */}
      <ProblemSection />

      {/* 5. How It Works */}
      <HowItWorksSection />

      {/* 6. Feature Section */}
      <FeatureSection />

      {/* 7. Live Analyzer Preview */}
      <LiveAnalyzerSection />

      {/* 8. BERT vs TF-IDF Section */}
      <ComparisonSection />

      {/* 9. Performance Preview */}
      <PerformancePreviewSection />

      {/* 10. Responsible AI / Limitations Section */}
      <ResponsibleAiSection />

      {/* 11. Model Card Preview */}
      <ModelCardPreviewSection />

      {/* 12. Final CTA */}
      <FinalCtaSection />
    </div>
  );
}
