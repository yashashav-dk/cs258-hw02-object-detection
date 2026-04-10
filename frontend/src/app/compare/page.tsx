"use client";

import { useState } from "react";
import Link from "next/link";
import FileUpload from "@/components/FileUpload";
import ComparisonTable from "@/components/ComparisonTable";
import { compareAll } from "@/lib/api";
import type { DetectionResponse } from "@/lib/api";

export default function ComparePage() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<DetectionResponse[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const responses = await compareAll(file);
      setResults(responses);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed");
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
            <Link href="/" className="text-gray-500 hover:text-gray-700">
              Detect
            </Link>
            <span className="text-blue-600 font-medium">Compare</span>
          </div>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-6">
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">
            Benchmark Comparison
          </h2>
          <p className="text-gray-600">
            Run the same image through all model+runtime combinations to compare
            latency, detection count, and speedup.
          </p>
        </div>

        <FileUpload
          onFileSelect={setFile}
          accept="image/jpeg,image/png"
          label="Upload image"
        />

        <button
          onClick={handleCompare}
          disabled={!file || loading}
          className="px-6 py-2 rounded-md bg-blue-600 text-white font-medium text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Running all combinations..." : "Compare All"}
        </button>

        {loading && (
          <div className="flex items-center gap-3 text-gray-600">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Loading and running each model+runtime combo...
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 text-red-700 text-sm">
            {error}
          </div>
        )}

        {results && results.length > 0 && <ComparisonTable responses={results} />}
        {results && results.length === 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 text-yellow-700 text-sm">
            No model+runtime combinations available. Run{" "}
            <code className="font-mono">scripts/export_models.py</code> first.
          </div>
        )}
      </main>
    </div>
  );
}
