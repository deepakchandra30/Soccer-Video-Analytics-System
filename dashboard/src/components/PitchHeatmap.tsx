import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useStore } from '../store';

const W = 700, H = 456; // 105:68 ratio scaled
const PAD = 20;
const PW = W - PAD * 2, PH = H - PAD * 2;

export const PitchHeatmap: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const { events } = useStore();

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g').attr('transform', `translate(${PAD},${PAD})`);

    // pitch background
    g.append('rect').attr('width', PW).attr('height', PH)
      .attr('fill', '#1a5c2a').attr('rx', 4);

    // pitch lines
    const line = (x1: number, y1: number, x2: number, y2: number) =>
      g.append('line').attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2)
        .attr('stroke', 'rgba(255,255,255,0.35)').attr('stroke-width', 1.5);

    // outline
    g.append('rect').attr('width', PW).attr('height', PH)
      .attr('fill', 'none').attr('stroke', 'rgba(255,255,255,0.5)').attr('stroke-width', 2);

    // center line
    line(PW / 2, 0, PW / 2, PH);

    // center circle
    g.append('circle').attr('cx', PW / 2).attr('cy', PH / 2).attr('r', PH * 0.135)
      .attr('fill', 'none').attr('stroke', 'rgba(255,255,255,0.35)').attr('stroke-width', 1.5);

    // center spot
    g.append('circle').attr('cx', PW / 2).attr('cy', PH / 2).attr('r', 3)
      .attr('fill', 'rgba(255,255,255,0.5)');

    // penalty areas
    const paW = PW * 16.5 / 105, paH = PH * 40.32 / 68;
    const paY = (PH - paH) / 2;
    g.append('rect').attr('x', 0).attr('y', paY).attr('width', paW).attr('height', paH)
      .attr('fill', 'none').attr('stroke', 'rgba(255,255,255,0.35)').attr('stroke-width', 1.5);
    g.append('rect').attr('x', PW - paW).attr('y', paY).attr('width', paW).attr('height', paH)
      .attr('fill', 'none').attr('stroke', 'rgba(255,255,255,0.35)').attr('stroke-width', 1.5);

    // goal areas
    const gaW = PW * 5.5 / 105, gaH = PH * 18.32 / 68;
    const gaY = (PH - gaH) / 2;
    g.append('rect').attr('x', 0).attr('y', gaY).attr('width', gaW).attr('height', gaH)
      .attr('fill', 'none').attr('stroke', 'rgba(255,255,255,0.35)').attr('stroke-width', 1.5);
    g.append('rect').attr('x', PW - gaW).attr('y', gaY).attr('width', gaW).attr('height', gaH)
      .attr('fill', 'none').attr('stroke', 'rgba(255,255,255,0.35)').attr('stroke-width', 1.5);

    // penalty spots
    const penX = PW * 11 / 105;
    g.append('circle').attr('cx', penX).attr('cy', PH / 2).attr('r', 3)
      .attr('fill', 'rgba(255,255,255,0.4)');
    g.append('circle').attr('cx', PW - penX).attr('cy', PH / 2).attr('r', 3)
      .attr('fill', 'rgba(255,255,255,0.4)');

    // plot events on pitch
    if (events.length > 0) {
      const eventColors: Record<string, string> = {
        'Goal': '#22c55e', 'Foul': '#f59e0b', 'Yellow card': '#eab308',
        'Red card': '#ef4444', 'Corner': '#3b82f6', 'Shots on target': '#f97316',
      };

      events.forEach((e, i) => {
        // distribute events along the timeline position on pitch
        const t = e.position / (50 * 60 * 1000); // normalize to ~50 min
        const x = e.half === 1 ? t * PW * 0.48 : PW * 0.52 + t * PW * 0.48;
        const y = PH * 0.2 + Math.sin(i * 2.7) * PH * 0.3 + PH * 0.15;
        const color = eventColors[e.label] || '#94a3b8';

        g.append('circle')
          .attr('cx', Math.min(Math.max(x, 10), PW - 10))
          .attr('cy', Math.min(Math.max(y, 10), PH - 10))
          .attr('r', e.label === 'Goal' ? 8 : 5)
          .attr('fill', color)
          .attr('opacity', e.visibility === 'visible' ? 0.85 : 0.35)
          .attr('stroke', 'white')
          .attr('stroke-width', e.label === 'Goal' ? 2 : 0.5);

        if (e.label === 'Goal') {
          g.append('text')
            .attr('x', Math.min(Math.max(x, 10), PW - 10))
            .attr('y', Math.min(Math.max(y, 10), PH - 10) - 12)
            .attr('text-anchor', 'middle')
            .attr('fill', 'white').attr('font-size', '10px').attr('font-weight', 'bold')
            .text(e.game_time || '');
        }
      });

      // legend
      const legendData = Array.from(new Set(events.map(e => e.label))).slice(0, 6);
      const lg = svg.append('g').attr('transform', `translate(${PAD + 5}, ${H - 12})`);
      legendData.forEach((label, i) => {
        const color = eventColors[label] || '#94a3b8';
        lg.append('circle').attr('cx', i * 100).attr('cy', 0).attr('r', 4).attr('fill', color);
        lg.append('text').attr('x', i * 100 + 8).attr('y', 4)
          .attr('fill', '#94a3b8').attr('font-size', '9px').text(label);
      });
    }
  }, [events]);

  return (
    <div className="pitch-heatmap">
      <div className="panel-header">
        <h3>Pitch Event Map</h3>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }} />
    </div>
  );
};
