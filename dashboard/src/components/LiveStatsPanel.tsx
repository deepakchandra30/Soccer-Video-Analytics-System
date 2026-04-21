import React from 'react';
import { Users, Cpu, Eye, Zap, Radio, Activity, Target, Gauge } from 'lucide-react';

interface LiveStatsPanelProps {
  isConnected: boolean;
  isProcessing: boolean;
  stats: {
    frames_processed: number;
    total_players_detected: number;
    avg_players_per_frame: number;
    avg_confidence: number;
    camera_cuts_detected: number;
    unique_track_ids: number;
    processing_fps: number;
  } | null;
  overlaySettings: {
    showBoundingBoxes: boolean;
    showTrackIds: boolean;
    showConfidence: boolean;
    showTrails: boolean;
  };
  onToggleBoundingBoxes: () => void;
  onToggleTrackIds: () => void;
  onToggleConfidence: () => void;
  onToggleTrails: () => void;
}

const styles = {
  panel: {
    background: '#111827',
    border: '1px solid #1e293b',
    borderRadius: '12px',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    color: '#e2e8f0',
  },
  statusBar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 14px',
    background: '#0c1220',
    borderRadius: '8px',
    border: '1px solid #1e293b',
  },
  statusIndicator: (active: boolean) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '12px',
    fontWeight: 600 as const,
    letterSpacing: '0.05em',
    color: active ? '#22c55e' : '#ef4444',
  }),
  statusDot: (active: boolean) => ({
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    background: active ? '#22c55e' : '#ef4444',
    boxShadow: active ? '0 0 8px #22c55e80' : '0 0 8px #ef444480',
  }),
  processingBadge: (active: boolean) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '11px',
    fontWeight: 600 as const,
    letterSpacing: '0.05em',
    color: active ? '#22d3ee' : '#64748b',
    padding: '4px 10px',
    borderRadius: '9999px',
    background: active ? '#22d3ee15' : '#64748b15',
    border: `1px solid ${active ? '#22d3ee30' : '#64748b30'}`,
  }),
  pulsingDot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: '#22d3ee',
    animation: 'pulse-dot 1.5s ease-in-out infinite',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '10px',
  },
  statCard: {
    background: '#0c1220',
    border: '1px solid #1e293b',
    borderRadius: '8px',
    padding: '14px',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '8px',
  },
  statHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '11px',
    fontWeight: 500 as const,
    color: '#94a3b8',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  },
  statValue: {
    fontSize: '22px',
    fontWeight: 700 as const,
    fontFamily: "'Fira Code', monospace",
    color: '#e2e8f0',
    lineHeight: 1,
  },
  sectionTitle: {
    fontSize: '13px',
    fontWeight: 600 as const,
    color: '#94a3b8',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    marginBottom: '4px',
  },
  overlayControls: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '8px',
  },
  toggleRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 12px',
    background: '#0c1220',
    borderRadius: '6px',
    border: '1px solid #1e293b',
    fontSize: '13px',
    color: '#e2e8f0',
  },
  toggleButton: (active: boolean) => ({
    width: '40px',
    height: '22px',
    borderRadius: '11px',
    border: 'none',
    cursor: 'pointer',
    background: active ? '#3b82f6' : '#334155',
    position: 'relative' as const,
    transition: 'background 0.2s',
    padding: 0,
  }),
  toggleKnob: (active: boolean) => ({
    width: '16px',
    height: '16px',
    borderRadius: '50%',
    background: '#ffffff',
    position: 'absolute' as const,
    top: '3px',
    left: active ? '21px' : '3px',
    transition: 'left 0.2s',
  }),
  placeholder: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
    gap: '12px',
    padding: '40px 20px',
    color: '#64748b',
    fontSize: '14px',
    textAlign: 'center' as const,
  },
};

const pulseKeyframes = `
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.8); }
}
`;

function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div style={styles.statCard}>
      <div style={styles.statHeader}>
        <Icon size={14} color={color} />
        <span>{label}</span>
      </div>
      <div style={styles.statValue}>{value}</div>
    </div>
  );
}

function Toggle({
  label,
  active,
  onToggle,
}: {
  label: string;
  active: boolean;
  onToggle: () => void;
}) {
  return (
    <div style={styles.toggleRow}>
      <span>{label}</span>
      <button
        style={styles.toggleButton(active)}
        onClick={onToggle}
        aria-label={`Toggle ${label}`}
      >
        <div style={styles.toggleKnob(active)} />
      </button>
    </div>
  );
}

export const LiveStatsPanel: React.FC<LiveStatsPanelProps> = ({
  isConnected,
  isProcessing,
  stats,
  overlaySettings,
  onToggleBoundingBoxes,
  onToggleTrackIds,
  onToggleConfidence,
  onToggleTrails,
}) => {
  return (
    <div style={styles.panel}>
      <style>{pulseKeyframes}</style>

      {/* Connection Status Bar */}
      <div style={styles.statusBar}>
        <div style={styles.statusIndicator(isConnected)}>
          <div style={styles.statusDot(isConnected)} />
          {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
        </div>
        <div style={styles.processingBadge(isProcessing)}>
          {isProcessing && <div style={styles.pulsingDot} />}
          {isProcessing ? 'ANALYZING' : 'PAUSED'}
        </div>
      </div>

      {/* Stats Grid or Placeholder */}
      {stats === null ? (
        <div style={styles.placeholder}>
          <Activity size={32} color="#64748b" />
          <span>Start real-time analysis to see live stats</span>
        </div>
      ) : (
        <>
          <div style={styles.statsGrid}>
            <StatCard
              icon={Cpu}
              label="Frames Processed"
              value={stats.frames_processed.toLocaleString()}
              color="#3b82f6"
            />
            <StatCard
              icon={Users}
              label="Players Detected"
              value={stats.total_players_detected.toLocaleString()}
              color="#22c55e"
            />
            <StatCard
              icon={Activity}
              label="Avg Players/Frame"
              value={stats.avg_players_per_frame.toFixed(1)}
              color="#a78bfa"
            />
            <StatCard
              icon={Target}
              label="Unique Tracks"
              value={stats.unique_track_ids.toLocaleString()}
              color="#f59e0b"
            />
            <StatCard
              icon={Eye}
              label="Avg Confidence"
              value={`${(stats.avg_confidence * 100).toFixed(1)}%`}
              color="#22d3ee"
            />
            <StatCard
              icon={Zap}
              label="Camera Cuts"
              value={stats.camera_cuts_detected.toLocaleString()}
              color="#ef4444"
            />
            <StatCard
              icon={Gauge}
              label="Processing FPS"
              value={stats.processing_fps.toFixed(1)}
              color="#3b82f6"
            />
            <StatCard
              icon={Radio}
              label="Detection Rate"
              value={
                stats.frames_processed > 0
                  ? `${((stats.total_players_detected / stats.frames_processed) * 100).toFixed(1)}%`
                  : '0.0%'
              }
              color="#22c55e"
            />
          </div>
        </>
      )}

      {/* Overlay Controls */}
      <div style={styles.overlayControls}>
        <div style={styles.sectionTitle}>Overlay Controls</div>
        <Toggle
          label="Bounding Boxes"
          active={overlaySettings.showBoundingBoxes}
          onToggle={onToggleBoundingBoxes}
        />
        <Toggle
          label="Track IDs"
          active={overlaySettings.showTrackIds}
          onToggle={onToggleTrackIds}
        />
        <Toggle
          label="Confidence"
          active={overlaySettings.showConfidence}
          onToggle={onToggleConfidence}
        />
        <Toggle
          label="Trails"
          active={overlaySettings.showTrails}
          onToggle={onToggleTrails}
        />
      </div>

    </div>
  );
};
