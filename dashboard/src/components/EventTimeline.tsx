import React, { useEffect } from 'react';
import { useStore } from '../store';
import { fetchEvents } from '../api';
import { Crosshair, AlertTriangle, Flag, CircleDot, Square, Zap } from 'lucide-react';

const EVENT_ICONS: Record<string, React.ReactNode> = {
  'Goal': <CircleDot size={14} color="#22c55e" />,
  'Foul': <AlertTriangle size={14} color="#f59e0b" />,
  'Yellow card': <Square size={14} color="#eab308" />,
  'Red card': <Square size={14} color="#ef4444" />,
  'Corner': <Flag size={14} color="#3b82f6" />,
  'Substitution': <Zap size={14} color="#a855f7" />,
};

const EVENT_COLORS: Record<string, string> = {
  'Goal': '#22c55e',
  'Foul': '#f59e0b',
  'Yellow card': '#eab308',
  'Red card': '#ef4444',
  'Corner': '#3b82f6',
  'Substitution': '#a855f7',
  'Ball out of play': '#64748b',
  'Clearance': '#06b6d4',
  'Shots on target': '#f97316',
  'Shots off target': '#fb923c',
};

export const EventTimeline: React.FC = () => {
  const { selectedMatch, events, setEvents, setCurrentTime } = useStore();

  useEffect(() => {
    if (!selectedMatch) return;
    let cancelled = false;
    fetchEvents(selectedMatch.id)
      .then(data => { if (!cancelled) setEvents(data); })
      .catch(err => { if (!cancelled) console.error(err); });
    return () => { cancelled = true; };
  }, [selectedMatch, setEvents]);

  if (!selectedMatch) return null;

  const formatTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;
  };

  const half1 = events.filter(e => e.half === 1);
  const half2 = events.filter(e => e.half === 2);

  const renderEvent = (e: typeof events[0], i: number) => {
    const color = EVENT_COLORS[e.label] || '#94a3b8';
    const icon = EVENT_ICONS[e.label] || <Crosshair size={14} color={color} />;
    const isVisible = e.visibility === 'visible';

    return (
      <div
        key={`${e.half}-${e.position}-${i}`}
        className={`event-row ${!isVisible ? 'dimmed' : ''}`}
        onClick={() => setCurrentTime(e.position)}
      >
        <span className="event-icon">{icon}</span>
        <span className="event-time" style={{ color }}>{e.game_time || formatTime(e.position)}</span>
        <span className="event-label">{e.label}</span>
        {e.team && <span className="event-team">{e.team}</span>}
      </div>
    );
  };

  return (
    <div className="event-timeline">
      <div className="panel-header">
        <h3>Match Events</h3>
        <span className="badge">{events.length}</span>
      </div>

      <div className="timeline-content">
        <div className="half-section">
          <div className="half-label">First Half</div>
          {half1.map(renderEvent)}
        </div>

        <div className="half-divider">HT</div>

        <div className="half-section">
          <div className="half-label">Second Half</div>
          {half2.map(renderEvent)}
        </div>
      </div>
    </div>
  );
};
