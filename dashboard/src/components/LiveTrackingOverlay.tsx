import React, { useRef, useEffect, useCallback } from 'react';

interface LiveTrackingOverlayProps {
  players: Array<{
    track_id: number;
    bbox: [number, number, number, number]; // x1, y1, x2, y2 in video pixels
    confidence: number;
  }>;
  videoWidth: number;
  videoHeight: number;
  showBoundingBoxes: boolean;
  showTrackIds: boolean;
  showConfidence: boolean;
  overlayOpacity: number;
  isCameraCut?: boolean;
}

const COLORS = [
  '#22c55e',
  '#3b82f6',
  '#f59e0b',
  '#ef4444',
  '#a855f7',
  '#06b6d4',
  '#f97316',
  '#ec4899',
  '#14b8a6',
  '#8b5cf6',
];

/**
 * Compute the actual rendered area of the video inside its container,
 * accounting for letterboxing (black bars) when the video aspect ratio
 * does not match the container aspect ratio.
 */
function computeVideoDisplayRect(
  containerW: number,
  containerH: number,
  videoW: number,
  videoH: number
): { offsetX: number; offsetY: number; renderW: number; renderH: number } {
  if (videoW === 0 || videoH === 0) {
    return { offsetX: 0, offsetY: 0, renderW: containerW, renderH: containerH };
  }

  const containerAspect = containerW / containerH;
  const videoAspect = videoW / videoH;

  let renderW: number;
  let renderH: number;

  if (videoAspect > containerAspect) {
    // Video is wider than container -- pillarboxed (black bars top/bottom)
    renderW = containerW;
    renderH = containerW / videoAspect;
  } else {
    // Video is taller than container -- letterboxed (black bars left/right)
    renderH = containerH;
    renderW = containerH * videoAspect;
  }

  const offsetX = (containerW - renderW) / 2;
  const offsetY = (containerH - renderH) / 2;

  return { offsetX, offsetY, renderW, renderH };
}

export const LiveTrackingOverlay: React.FC<LiveTrackingOverlayProps> = ({
  players,
  videoWidth,
  videoHeight,
  showBoundingBoxes,
  showTrackIds,
  showConfidence,
  overlayOpacity,
  isCameraCut = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const canvasW = canvas.width;   // internal resolution (already DPR-scaled)
    const canvasH = canvas.height;

    ctx.clearRect(0, 0, canvasW, canvasH);

    // Guard: nothing to draw when video dimensions are unknown
    if (videoWidth === 0 || videoHeight === 0) return;

    // Guard: if players array is empty, just leave the canvas cleared
    if (players.length === 0 && !isCameraCut) return;

    // Compute where the video actually renders inside the container (letterbox-aware)
    const cssW = container.clientWidth;
    const cssH = container.clientHeight;
    const { offsetX, offsetY, renderW, renderH } = computeVideoDisplayRect(
      cssW,
      cssH,
      videoWidth,
      videoHeight
    );

    // Scale from CSS pixels to internal canvas pixels, then from video coords
    // to the rendered video area.
    const scaleX = (renderW * dpr) / videoWidth;
    const scaleY = (renderH * dpr) / videoHeight;
    const ox = offsetX * dpr;
    const oy = offsetY * dpr;

    ctx.globalAlpha = overlayOpacity;

    // Draw camera cut banner (spans full canvas width)
    if (isCameraCut) {
      ctx.save();
      ctx.globalAlpha = overlayOpacity * 0.85;
      ctx.fillStyle = '#ef4444';
      const bannerHeight = 36 * dpr;
      ctx.fillRect(0, 0, canvasW, bannerHeight);
      ctx.globalAlpha = overlayOpacity;
      ctx.fillStyle = '#ffffff';
      ctx.font = `bold ${16 * dpr}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('CAMERA CUT', canvasW / 2, bannerHeight / 2);
      ctx.restore();
    }

    for (const player of players) {
      const { track_id, bbox, confidence } = player;
      const [x1, y1, x2, y2] = bbox;

      const sx1 = ox + x1 * scaleX;
      const sy1 = oy + y1 * scaleY;
      const sx2 = ox + x2 * scaleX;
      const sy2 = oy + y2 * scaleY;
      const boxW = sx2 - sx1;
      const boxH = sy2 - sy1;

      const color = COLORS[track_id % COLORS.length];

      // Bounding box
      if (showBoundingBoxes) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2 * dpr;
        ctx.strokeRect(sx1, sy1, boxW, boxH);
      }

      // Track ID pill
      if (showTrackIds) {
        const label = `P${track_id}`;
        ctx.font = `bold ${12 * dpr}px sans-serif`;
        const textMetrics = ctx.measureText(label);
        const pillW = textMetrics.width + 10 * dpr;
        const pillH = 18 * dpr;
        const pillX = sx1;
        const pillY = sy1 - pillH - 2 * dpr;

        ctx.fillStyle = color;
        ctx.beginPath();
        const radius = 4 * dpr;
        ctx.moveTo(pillX + radius, pillY);
        ctx.lineTo(pillX + pillW - radius, pillY);
        ctx.arcTo(pillX + pillW, pillY, pillX + pillW, pillY + radius, radius);
        ctx.lineTo(pillX + pillW, pillY + pillH - radius);
        ctx.arcTo(pillX + pillW, pillY + pillH, pillX + pillW - radius, pillY + pillH, radius);
        ctx.lineTo(pillX + radius, pillY + pillH);
        ctx.arcTo(pillX, pillY + pillH, pillX, pillY + pillH - radius, radius);
        ctx.lineTo(pillX, pillY + radius);
        ctx.arcTo(pillX, pillY, pillX + radius, pillY, radius);
        ctx.closePath();
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, pillX + pillW / 2, pillY + pillH / 2);
      }

      // Confidence text
      if (showConfidence) {
        const confText = `${(confidence * 100).toFixed(1)}%`;
        ctx.font = `${11 * dpr}px sans-serif`;
        ctx.fillStyle = color;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(confText, sx1, sy2 + 3 * dpr);
      }
    }
  }, [players, videoWidth, videoHeight, showBoundingBoxes, showTrackIds, showConfidence, overlayOpacity, isCameraCut]);

  // Resize canvas to match container, respecting devicePixelRatio
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        const dpr = window.devicePixelRatio || 1;

        // Internal (buffer) resolution -- scaled for sharp rendering on HiDPI
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);

        // CSS display size -- matches the container exactly
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;

        draw();
      }
    });

    observer.observe(container);

    return () => {
      observer.disconnect();
    };
  }, [draw]);

  // Redraw when players or settings change
  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
        }}
      />
    </div>
  );
};
