"use client";

import type { DetectionResponse } from "@/lib/api";
import { getVideoResultUrl } from "@/lib/api";

interface VideoResultProps {
  response: DetectionResponse;
}

export default function VideoResult({ response }: VideoResultProps) {
  const totalFrames = response.usage.total_frames || 0;
  const avgLatency = response.usage.avg_latency_per_frame_ms || 0;
  const totalDetections = response.results.reduce(
    (sum, fr) => sum + fr.detections.length,
    0
  );

  const videoUrl = response.annotated_video_url
    ? getVideoResultUrl(response.annotated_video_url)
    : null;

  // Per-frame latency data for bar chart
  const frameTimes = response.results.map((_, i) => i);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center rounded-full bg-blue-100 px-3 py-1 text-sm font-medium text-blue-800">
          {response.usage.inference_time_ms.toFixed(1)} ms total
        </span>
        <span className="inline-flex items-center rounded-full bg-purple-100 px-3 py-1 text-sm font-medium text-purple-800">
          {avgLatency.toFixed(1)} ms/frame avg
        </span>
        <span className="inline-flex items-center rounded-full bg-green-100 px-3 py-1 text-sm font-medium text-green-800">
          {totalFrames} frames
        </span>
        <span className="inline-flex items-center rounded-full bg-orange-100 px-3 py-1 text-sm font-medium text-orange-800">
          {totalDetections} total detections
        </span>
        <span className="text-sm text-gray-500">
          {response.model} / {response.runtime}
        </span>
      </div>

      {videoUrl && (
        <video
          src={videoUrl}
          controls
          className="max-w-full rounded"
        >
          Your browser does not support the video tag.
        </video>
      )}

      {/* Per-frame latency chart */}
      {response.results.length > 1 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">
            Per-frame Detections
          </h3>
          <div className="flex items-end gap-px h-32">
            {response.results.map((fr, i) => {
              const maxDets = Math.max(
                ...response.results.map((r) => r.detections.length),
                1
              );
              const height = (fr.detections.length / maxDets) * 100;
              return (
                <div
                  key={i}
                  className="bg-blue-400 hover:bg-blue-600 transition-colors flex-1 min-w-[2px] rounded-t"
                  style={{ height: `${Math.max(height, 2)}%` }}
                  title={`Frame ${i}: ${fr.detections.length} detections`}
                />
              );
            })}
          </div>
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>Frame 0</span>
            <span>Frame {response.results.length - 1}</span>
          </div>
        </div>
      )}
    </div>
  );
}
