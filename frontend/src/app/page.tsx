"use client";

import { useState } from "react";
import Link from "next/link";
import FileUpload from "@/components/FileUpload";
import ModelSelector from "@/components/ModelSelector";
import DetectionResult from "@/components/DetectionResult";
import VideoResult from "@/components/VideoResult";
import { detectImage, detectVideo, getVideoResultUrl } from "@/lib/api";
import type { DetectionResponse } from "@/lib/api";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState("yolov8m");
  const [runtime, setRuntime] = useState("pytorch");
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isVideo, setIsVideo] = useState(false);

  const handleFileSelect = (f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    const video = f.type.startsWith("video/");
    setIsVideo(video);
    if (!video) {
      setImageSrc(URL.createObjectURL(f));
    } else {
      setImageSrc(null);
    }
  };

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let response: DetectionResponse;
      if (isVideo) {
        response = await detectVideo(file, model, runtime);
      } else {
        response = await detectImage(file, model, runtime);
      }
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <h1 className="text-lg font-semibold text-gray-900">
            Object Detection
          </h1>
          <div className="flex gap-4 text-sm">
            <span className="text-blue-600 font-medium">Detect</span>
            <Link href="/compare" className="text-gray-500 hover:text-gray-700">
              Compare
            </Link>
          </div>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-6">
        <FileUpload onFileSelect={handleFileSelect} />

        <div className="flex items-end gap-4">
          <ModelSelector
            model={model}
            runtime={runtime}
            onModelChange={setModel}
            onRuntimeChange={setRuntime}
          />
          <button
            onClick={handleDetect}
            disabled={!file || loading}
            className="px-6 py-2 rounded-md bg-blue-600 text-white font-medium text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Detecting..." : "Detect"}
          </button>
        </div>

        {loading && (
          <div className="flex items-center gap-3 text-gray-600">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            {isVideo ? "Processing video frames..." : "Running inference..."}
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 text-red-700 text-sm">
            {error}
          </div>
        )}

        {result && !isVideo && imageSrc && (
          <DetectionResult response={result} imageSrc={imageSrc} />
        )}

        {result && isVideo && (
          <VideoResult response={result} />
        )}
      </main>
    </div>
  );
}
