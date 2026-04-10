"use client";

import { useEffect, useRef } from "react";
import type { Detection } from "@/lib/api";

interface BBoxOverlayProps {
  imageSrc: string;
  detections: Detection[];
}

const CLASS_COLORS: Record<string, string> = {};
const PALETTE = [
  "#ef4444", "#3b82f6", "#22c55e", "#eab308", "#a855f7",
  "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#8b5cf6",
];

function getColor(className: string): string {
  if (!CLASS_COLORS[className]) {
    CLASS_COLORS[className] = PALETTE[Object.keys(CLASS_COLORS).length % PALETTE.length];
  }
  return CLASS_COLORS[className];
}

export default function BBoxOverlay({ imageSrc, detections }: BBoxOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const draw = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(img, 0, 0);

      for (const det of detections) {
        const [x1, y1, x2, y2] = det.bbox;
        const color = getColor(det.class);
        const w = x2 - x1;
        const h = y2 - y1;

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, Math.min(img.naturalWidth, img.naturalHeight) / 300);
        ctx.strokeRect(x1, y1, w, h);

        // Label background
        const label = `${det.class} ${(det.confidence * 100).toFixed(1)}%`;
        const fontSize = Math.max(12, Math.min(img.naturalWidth, img.naturalHeight) / 50);
        ctx.font = `bold ${fontSize}px sans-serif`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - fontSize - 4, textWidth + 8, fontSize + 4);

        // Label text
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x1 + 4, y1 - 4);
      }
    };

    if (img.complete) draw();
    else img.onload = draw;
  }, [imageSrc, detections]);

  return (
    <div className="relative">
      <img ref={imgRef} src={imageSrc} alt="" className="hidden" crossOrigin="anonymous" />
      <canvas ref={canvasRef} className="max-w-full h-auto rounded" />
    </div>
  );
}
