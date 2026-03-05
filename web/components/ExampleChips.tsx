"use client";

interface ExampleChipsProps {
  onSelect: (query: string) => void;
}

const EXAMPLES = [
  "What are the top-performing QSR campaigns?",
  "Compare Dunkin' Q3 vs Q4 performance",
  "Generate a report for the best ROAS campaign",
  "Which campaigns had the highest visit lift?",
  "Recommend audience segments for a CPG back-to-school campaign",
  "Show me all active automotive campaigns",
];

export default function ExampleChips({ onSelect }: ExampleChipsProps) {
  return (
    <div className="flex flex-wrap justify-center gap-2 max-w-lg">
      {EXAMPLES.map((example, i) => (
        <button
          key={i}
          onClick={() => onSelect(example)}
          className="text-sm px-3 py-1.5 rounded-full border border-gray-300 text-gray-600 hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700 transition-colors"
        >
          {example}
        </button>
      ))}
    </div>
  );
}
