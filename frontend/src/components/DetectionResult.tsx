"use client";

import type { DetectionResponse } from "@/lib/api";
import BBoxOverlay from "./BBoxOverlay";

interface DetectionResultProps {
  response: DetectionResponse;
  imageSrc: string;
}

export default function DetectionResult({ response, imageSrc }: DetectionResultProps) {
  const detections = response.results[0]?.detections || [];
  const latency = response.usage.inference_time_ms;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center rounded-full bg-blue-100 px-3 py-1 text-sm font-medium text-blue-800">
          {latency.toFixed(1)} ms
        </span>
        <span className="inline-flex items-center rounded-full bg-green-100 px-3 py-1 text-sm font-medium text-green-800">
          {detections.length} detection{detections.length !== 1 ? "s" : ""}
        </span>
        <span className="text-sm text-gray-500">
          {response.model} / {response.runtime}
        </span>
      </div>
      <BBoxOverlay imageSrc={imageSrc} detections={detections} />
    </div>
  );
}
