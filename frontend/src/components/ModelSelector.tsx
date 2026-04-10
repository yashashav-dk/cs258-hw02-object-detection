"use client";

interface ModelSelectorProps {
  model: string;
  runtime: string;
  onModelChange: (model: string) => void;
  onRuntimeChange: (runtime: string) => void;
}

const MODELS = ["yolov8m", "yolov11m"];
const RUNTIMES = ["pytorch", "onnx", "tensorrt"];

export default function ModelSelector({
  model,
  runtime,
  onModelChange,
  onRuntimeChange,
}: ModelSelectorProps) {
  return (
    <div className="flex gap-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Model
        </label>
        <select
          value={model}
          onChange={(e) => onModelChange(e.target.value)}
          className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        >
          {MODELS.map((m) => (
            <option key={m} value={m}>
              {m.toUpperCase()}
            </option>
          ))}
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Runtime
        </label>
        <select
          value={runtime}
          onChange={(e) => onRuntimeChange(e.target.value)}
          className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        >
          {RUNTIMES.map((r) => (
            <option key={r} value={r}>
              {r.charAt(0).toUpperCase() + r.slice(1)}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
