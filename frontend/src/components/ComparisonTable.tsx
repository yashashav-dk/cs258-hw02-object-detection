"use client";

import type { DetectionResponse } from "@/lib/api";

interface ComparisonTableProps {
  responses: DetectionResponse[];
}

export default function ComparisonTable({ responses }: ComparisonTableProps) {
  // Compute baseline latency per model (PyTorch)
  const baselines: Record<string, number> = {};
  for (const r of responses) {
    if (r.runtime === "pytorch") {
      baselines[r.model] = r.usage.inference_time_ms;
    }
  }

  // Find fastest overall runtime
  const fastest = responses.reduce(
    (min, r) => (r.usage.inference_time_ms < min.usage.inference_time_ms ? r : min),
    responses[0]
  );

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="px-4 py-3 text-left font-medium text-gray-700">Model</th>
            <th className="px-4 py-3 text-left font-medium text-gray-700">Runtime</th>
            <th className="px-4 py-3 text-right font-medium text-gray-700">Latency (ms)</th>
            <th className="px-4 py-3 text-right font-medium text-gray-700">Detections</th>
            <th className="px-4 py-3 text-right font-medium text-gray-700">Speedup</th>
          </tr>
        </thead>
        <tbody>
          {responses.map((r, i) => {
            const baseline = baselines[r.model];
            const speedup = baseline
              ? (baseline / r.usage.inference_time_ms).toFixed(2) + "x"
              : "—";
            const detCount = r.results[0]?.detections.length || 0;
            const isFastest = r === fastest;

            return (
              <tr
                key={`${r.model}-${r.runtime}-${i}`}
                className={`border-b border-gray-100 ${
                  isFastest ? "bg-green-50" : "hover:bg-gray-50"
                }`}
              >
                <td className="px-4 py-3 font-mono text-gray-900">{r.model}</td>
                <td className="px-4 py-3 font-mono text-gray-700">{r.runtime}</td>
                <td className="px-4 py-3 text-right font-mono text-gray-900">
                  {r.usage.inference_time_ms.toFixed(2)}
                  {isFastest && (
                    <span className="ml-2 text-xs text-green-700 font-medium">
                      ⚡ fastest
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-right text-gray-700">{detCount}</td>
                <td className="px-4 py-3 text-right font-mono text-gray-700">
                  {speedup}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
