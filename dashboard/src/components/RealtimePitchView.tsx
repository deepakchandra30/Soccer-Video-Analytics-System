import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface RealtimePitchViewProps {
  players: Array<{
    track_id: number;
    bbox: [number, number, number, number]; // x1,y1,x2,y2
    confidence: number;
  }>;
  videoWidth: number;
  videoHeight: number;
  isProcessing: boolean;
  stats: {
    frames_processed: number;
    unique_track_ids: number;
    avg_confidence: number;
    processing_fps?: number;
  } | null;
}

interface MappedPlayer {
  track_id: number;
  x: number;
  y: number;
  confidence: number;
}

const W = 500, H = 325;
const PAD = 20;
const PW = W - PAD * 2, PH = H - PAD * 2;

const PLAYER_COLORS = [
  '#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#a855f7',
  '#84cc16', '#e11d48', '#0ea5e9', '#d946ef', '#10b981',
  '#fbbf24', '#6366f1', '#fb923c', '#2dd4bf', '#f43f5e',
];

function getPlayerColor(trackId: number): string {
  return PLAYER_COLORS[trackId % PLAYER_COLORS.length];
}

export const RealtimePitchView: React.FC<RealtimePitchViewProps> = ({
  players,
  videoWidth,
  videoHeight,
  isProcessing,
  stats,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const pitchDrawn = useRef(false);

  const hasActivity = players.length > 0 || stats != null;

  // Draw the static pitch lines once
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    pitchDrawn.current = true;

    const g = svg.append('g')
      .attr('transform', `translate(${PAD},${PAD})`)
      .attr('class', 'pitch-layer');

    const lineColor = 'rgba(255,255,255,0.4)';
    const lineWidth = 1.5;

    // Pitch background
    g.append('rect')
      .attr('width', PW).attr('height', PH)
      .attr('fill', '#1a5c2a').attr('rx', 4);

    // Pitch outline
    g.append('rect')
      .attr('width', PW).attr('height', PH)
      .attr('fill', 'none')
      .attr('stroke', lineColor).attr('stroke-width', 2);

    // Center line
    g.append('line')
      .attr('x1', PW / 2).attr('y1', 0)
      .attr('x2', PW / 2).attr('y2', PH)
      .attr('stroke', lineColor).attr('stroke-width', lineWidth);

    // Center circle
    g.append('circle')
      .attr('cx', PW / 2).attr('cy', PH / 2)
      .attr('r', PH * 0.135)
      .attr('fill', 'none')
      .attr('stroke', lineColor).attr('stroke-width', lineWidth);

    // Center spot
    g.append('circle')
      .attr('cx', PW / 2).attr('cy', PH / 2)
      .attr('r', 3)
      .attr('fill', lineColor);

    // Penalty areas
    const paW = PW * 16.5 / 105;
    const paH = PH * 40.32 / 68;
    const paY = (PH - paH) / 2;

    g.append('rect')
      .attr('x', 0).attr('y', paY)
      .attr('width', paW).attr('height', paH)
      .attr('fill', 'none')
      .attr('stroke', lineColor).attr('stroke-width', lineWidth);
    g.append('rect')
      .attr('x', PW - paW).attr('y', paY)
      .attr('width', paW).attr('height', paH)
      .attr('fill', 'none')
      .attr('stroke', lineColor).attr('stroke-width', lineWidth);

    // Goal areas
    const gaW = PW * 5.5 / 105;
    const gaH = PH * 18.32 / 68;
    const gaY = (PH - gaH) / 2;

    g.append('rect')
      .attr('x', 0).attr('y', gaY)
      .attr('width', gaW).attr('height', gaH)
      .attr('fill', 'none')
      .attr('stroke', lineColor).attr('stroke-width', lineWidth);
    g.append('rect')
      .attr('x', PW - gaW).attr('y', gaY)
      .attr('width', gaW).attr('height', gaH)
      .attr('fill', 'none')
      .attr('stroke', lineColor).attr('stroke-width', lineWidth);

    // Penalty spots
    const penX = PW * 11 / 105;
    g.append('circle')
      .attr('cx', penX).attr('cy', PH / 2)
      .attr('r', 3).attr('fill', lineColor);
    g.append('circle')
      .attr('cx', PW - penX).attr('cy', PH / 2)
      .attr('r', 3).attr('fill', lineColor);

    // Player layer (rendered on top of pitch lines)
    svg.append('g')
      .attr('transform', `translate(${PAD},${PAD})`)
      .attr('class', 'player-layer');
  }, []);

  // Update player positions with D3 data join and transitions
  useEffect(() => {
    if (!svgRef.current || !pitchDrawn.current) return;

    const svg = d3.select(svgRef.current);
    const playerLayer = svg.select<SVGGElement>('.player-layer');
    if (playerLayer.empty()) return;

    const vw = videoWidth || 1;
    const vh = videoHeight || 1;

    const mappedPlayers: MappedPlayer[] = players.map((p) => {
      const cx = (p.bbox[0] + p.bbox[2]) / 2;
      const cy = (p.bbox[1] + p.bbox[3]) / 2;
      return {
        track_id: p.track_id,
        x: (cx / vw) * PW,
        y: (cy / vh) * PH,
        confidence: p.confidence,
      };
    });

    // Data join on track_id
    const dots = playerLayer
      .selectAll<SVGGElement, MappedPlayer>('.player-dot')
      .data(mappedPlayers, (d: MappedPlayer) => String(d.track_id));

    // EXIT
    dots.exit()
      .transition().duration(150)
      .attr('opacity', 0)
      .remove();

    // ENTER
    const enter = dots.enter()
      .append('g')
      .attr('class', 'player-dot')
      .attr('transform', (d) => `translate(${d.x},${d.y})`)
      .attr('opacity', 0);

    enter.append('circle')
      .attr('r', 6)
      .attr('fill', (d) => getPlayerColor(d.track_id))
      .attr('stroke', 'white')
      .attr('stroke-width', 1);

    enter.append('text')
      .attr('x', 8)
      .attr('y', 3)
      .attr('fill', 'white')
      .attr('font-size', '9px')
      .attr('font-weight', '600')
      .attr('pointer-events', 'none')
      .text((d) => String(d.track_id));

    enter.transition().duration(200).attr('opacity', 1);

    // UPDATE
    const merged = enter.merge(dots);

    merged
      .transition().duration(200)
      .attr('transform', (d) => `translate(${d.x},${d.y})`);

    merged.select('circle')
      .attr('fill', (d) => getPlayerColor(d.track_id));

    merged.select('text')
      .text((d) => String(d.track_id));
  }, [players, videoWidth, videoHeight]);

  return (
    <div
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: '12px',
        overflow: 'hidden',
      }}
    >
      <div className="panel-header" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <h3 style={{ margin: 0 }}>Live Pitch Tracking</h3>
        <span
          style={{
            display: 'inline-block',
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: hasActivity ? '#22c55e' : '#64748b',
            animation: hasActivity ? 'pulse-dot 1.5s ease-in-out infinite' : 'none',
          }}
        />
      </div>

      <div style={{ position: 'relative' }}>
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} />

        {hasActivity && stats ? (
          <div
            style={{
              position: 'absolute',
              top: '8px',
              right: '8px',
              background: 'rgba(0,0,0,0.7)',
              borderRadius: '8px',
              padding: '8px 12px',
              color: 'white',
              fontSize: '11px',
              lineHeight: '1.6',
              pointerEvents: 'none',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <span
                style={{
                  display: 'inline-block',
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  backgroundColor: '#22c55e',
                  animation: 'pulse-dot 1.5s ease-in-out infinite',
                }}
              />
              <span style={{ fontWeight: 700, color: '#22c55e', letterSpacing: '0.05em' }}>LIVE</span>
            </div>
            <div style={{ color: '#94a3b8' }}>
              Frames: <span style={{ color: 'white' }}>{stats.frames_processed}</span>
            </div>
            <div style={{ color: '#94a3b8' }}>
              Players: <span style={{ color: 'white' }}>{stats.unique_track_ids}</span>
            </div>
            <div style={{ color: '#94a3b8' }}>
              FPS: <span style={{ color: 'white' }}>
                {stats.processing_fps != null ? stats.processing_fps.toFixed(1) : '--'}
              </span>
            </div>
          </div>
        ) : !hasActivity ? (
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'rgba(0,0,0,0.35)',
              color: '#94a3b8',
              fontSize: '14px',
              fontWeight: 600,
              letterSpacing: '0.03em',
            }}
          >
            Not Active
          </div>
        ) : null}
      </div>
    </div>
  );
};
