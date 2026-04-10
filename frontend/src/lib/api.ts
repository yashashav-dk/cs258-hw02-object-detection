const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Detection {
  bbox: number[];
  class: string;
  confidence: number;
}

export interface FrameResult {
  frame: number;
  detections: Detection[];
}

export interface UsageInfo {
  inference_time_ms: number;
  avg_latency_per_frame_ms?: number;
  total_frames?: number;
}

export interface DetectionResponse {
  id: string;
  object: string;
  model: string;
  runtime: string;
  created: number;
  usage: UsageInfo;
  results: FrameResult[];
  annotated_video_url?: string;
}

export interface RuntimeInfo {
  runtime: string;
  available: boolean;
  loaded: boolean;
}

export interface ModelInfo {
  id: string;
  object: string;
  runtimes: RuntimeInfo[];
}

export interface ModelsResponse {
  object: string;
  data: ModelInfo[];
}

export async function getModels(): Promise<ModelsResponse> {
  const res = await fetch(`${API_URL}/v1/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  return res.json();
}

export async function detectImage(
  file: File,
  model: string,
  runtime: string
): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model", model);
  formData.append("runtime", runtime);

  const res = await fetch(`${API_URL}/v1/detect/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function detectVideo(
  file: File,
  model: string,
  runtime: string
): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model", model);
  formData.append("runtime", runtime);

  const res = await fetch(`${API_URL}/v1/detect/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function compareAll(file: File): Promise<DetectionResponse[]> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/v1/detect/compare`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export function getVideoResultUrl(path: string): string {
  return `${API_URL}${path}`;
}
