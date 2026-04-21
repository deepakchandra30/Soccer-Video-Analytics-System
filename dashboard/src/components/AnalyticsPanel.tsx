import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useStore } from '../store';
import { fetchAnalytics } from '../api';
import { BarChart3, Clock, Eye, Layers, TrendingUp, Activity } from 'lucide-react';

// animated counter hook
function useAnimatedValue(target: number, duration = 800) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let rafId: number;
    const startTime = performance.now();
    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(target * eased));
      if (progress < 1) rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [target, duration]);
  return value;
}

const StatCard: React.FC<{ icon: React.ReactNode; value: number; label: string; suffix?: string; color?: string }> =
  ({ icon, value, label, suffix = '', color }) => {
  const animated = useAnimatedValue(value);
  return (
    <div className="stat-card">
      {icon}
      <div className="stat-value" style={color ? { color } : undefined}>{animated}{suffix}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
};

export const AnalyticsPanel: React.FC = () => {
  const { selectedMatch, analytics, setAnalytics } = useStore();
  const chartRef = useRef<SVGSVGElement>(null);
  const pieRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!selectedMatch) return;
    let cancelled = false;
    fetchAnalytics(selectedMatch.id)
      .then(data => { if (!cancelled) setAnalytics(data); })
      .catch(err => { if (!cancelled) console.error(err); });
    return () => { cancelled = true; };
  }, [selectedMatch, setAnalytics]);

  // draw event distribution bar chart with gradients
  useEffect(() => {
    if (!analytics || !chartRef.current) return;
    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove();

    const counts = analytics.event_counts;
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
    if (entries.length === 0) return;

    const w = 380, h = 180;
    const margin = { top: 10, right: 10, bottom: 60, left: 40 };
    const iw = w - margin.left - margin.right;
    const ih = h - margin.top - margin.bottom;

    // add gradient definitions
    const defs = svg.append('defs');
    const colors: Record<string, string> = {
      'Goal': '#22c55e', 'Foul': '#f59e0b', 'Yellow card': '#eab308',
      'Red card': '#ef4444', 'Corner': '#3b82f6', 'Substitution': '#a855f7',
      'Ball out of play': '#64748b', 'Clearance': '#06b6d4',
      'Shots on target': '#f97316', 'Shots off target': '#fb923c',
      'Offside': '#8b5cf6',
    };

    entries.forEach(([label]) => {
      const color = colors[label] || '#6366f1';
      const grad = defs.append('linearGradient')
        .attr('id', `bar-${label.replace(/\s+/g, '-')}`)
        .attr('x1', '0').attr('y1', '0').attr('x2', '0').attr('y2', '1');
      grad.append('stop').attr('offset', '0%').attr('stop-color', color).attr('stop-opacity', 1);
      grad.append('stop').attr('offset', '100%').attr('stop-color', color).attr('stop-opacity', 0.4);
    });

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().domain(entries.map(d => d[0])).range([0, iw]).padding(0.3);
    const y = d3.scaleLinear().domain([0, d3.max(entries, d => d[1]) || 1]).range([ih, 0]);

    // bars with gradient fill and glow
    g.selectAll('rect').data(entries).enter().append('rect')
      .attr('x', d => x(d[0]) || 0)
      .attr('y', ih)
      .attr('width', x.bandwidth())
      .attr('height', 0)
      .attr('rx', 4)
      .attr('fill', d => `url(#bar-${d[0].replace(/\s+/g, '-')})`)
      .transition().duration(600).delay((_, i) => i * 60)
      .attr('y', d => y(d[1]))
      .attr('height', d => ih - y(d[1]));

    // value labels
    g.selectAll('.bar-label').data(entries).enter().append('text')
      .attr('x', d => (x(d[0]) || 0) + x.bandwidth() / 2)
      .attr('y', d => y(d[1]) - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e2e8f0').attr('font-size', '11px').attr('font-weight', '500')
      .attr('font-family', 'Fira Code, monospace')
      .attr('opacity', 0)
      .text(d => d[1])
      .transition().duration(400).delay((_, i) => i * 60 + 300)
      .attr('opacity', 1);

    // x axis
    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).tickSize(0))
      .selectAll('text')
      .attr('transform', 'rotate(-35)').style('text-anchor', 'end')
      .attr('fill', '#94a3b8').attr('font-size', '10px');

    g.selectAll('.domain').attr('stroke', '#334155');
    g.selectAll('.tick line').remove();

  }, [analytics]);

  // draw donut chart
  useEffect(() => {
    if (!analytics || !pieRef.current) return;
    const svg = d3.select(pieRef.current);
    svg.selectAll('*').remove();

    const data = [
      { label: 'H1', value: analytics.half1_events, color: '#3b82f6' },
      { label: 'H2', value: analytics.half2_events, color: '#a855f7' },
    ];

    const size = 120, radius = size / 2 - 10;
    const g = svg.append('g').attr('transform', `translate(${size/2},${size/2})`);

    const pie = d3.pie<typeof data[0]>().value(d => d.value).sort(null);
    const arc = d3.arc<d3.PieArcDatum<typeof data[0]>>().innerRadius(radius * 0.6).outerRadius(radius);

    g.selectAll('path').data(pie(data)).enter().append('path')
      .attr('d', arc as any)
      .attr('fill', d => d.data.color)
      .attr('opacity', 0.85)
      .attr('stroke', '#111827')
      .attr('stroke-width', 2);

    g.selectAll('text').data(pie(data)).enter().append('text')
      .attr('transform', d => `translate(${arc.centroid(d as any)})`)
      .attr('text-anchor', 'middle').attr('fill', 'white').attr('font-size', '11px')
      .attr('font-family', 'Fira Code, monospace').attr('font-weight', '500')
      .text(d => `${d.data.label}: ${d.data.value}`);

    // center label
    g.append('text').attr('text-anchor', 'middle').attr('fill', '#e2e8f0')
      .attr('font-size', '14px').attr('font-weight', '600')
      .attr('font-family', 'Fira Code, monospace').attr('dy', '0.35em')
      .text(analytics.half1_events + analytics.half2_events);
  }, [analytics]);

  if (!selectedMatch) return null;
  if (!analytics) return (
    <div className="analytics-panel">
      <div className="panel-header"><h3><Activity size={16} /> Match Analytics</h3></div>
      <div className="empty-state">Loading analytics...</div>
    </div>
  );

  return (
    <div className="analytics-panel">
      <div className="panel-header">
        <h3><BarChart3 size={16} /> Match Analytics</h3>
        <span className="badge"><TrendingUp size={12} /> Real-time</span>
      </div>

      <div className="stats-grid">
        <StatCard icon={<Layers size={20} />} value={analytics.total_events} label="Total Events" />
        <StatCard icon={<Eye size={20} />} value={analytics.visible_events} label="Visible" color="#22c55e" />
        <StatCard icon={<Clock size={20} />} value={Math.round(analytics.feature_stats.duration_min)} label="Duration" suffix="m" color="#22d3ee" />
        <StatCard icon={<Activity size={20} />} value={analytics.feature_stats.total_frames} label="Frames" color="#a855f7" />
      </div>

      <div className="chart-row">
        <div className="chart-container">
          <h4>Event Distribution</h4>
          <svg ref={chartRef} viewBox="0 0 380 180" style={{ width: '100%' }} />
        </div>
        <div className="chart-container small">
          <h4>Half Split</h4>
          <svg ref={pieRef} viewBox="0 0 120 120" style={{ width: '120px', margin: '0 auto', display: 'block' }} />
        </div>
      </div>

      {/* team breakdown */}
      {analytics.team_events && Object.keys(analytics.team_events).length > 1 && (
        <div className="team-breakdown">
          <h4>By Team</h4>
          {Object.entries(analytics.team_events).map(([team, counts]) => (
            <div key={team} className="team-row">
              <span className="team-name">{team || 'Unknown'}</span>
              <div className="team-events">
                {Object.entries(counts).map(([label, count]) => (
                  <span key={label} className="team-event-badge">{label}: {count}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
