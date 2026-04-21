import { useEffect, useState } from 'react';
import { MatchSelector } from './components/MatchSelector';
import { VideoPlayer } from './components/VideoPlayer';
import { EventTimeline } from './components/EventTimeline';
import { AnalyticsPanel } from './components/AnalyticsPanel';
import { PitchHeatmap } from './components/PitchHeatmap';
import { RealtimePitchView } from './components/RealtimePitchView';
import { LiveStatsPanel } from './components/LiveStatsPanel';
import { useStore } from './store';
import { useRealtimeStore } from './realtimeStore';
import { fetchHealth } from './api';
import './App.css';

function App() {
  const { matches, selectedMatch } = useStore();
  const {
    isConnected, currentFrame, stats, videoInfo,
    showBoundingBoxes, showTrackIds, showConfidence, showTrails,
    toggleBoundingBoxes, toggleTrackIds, toggleConfidence, toggleTrails,
  } = useRealtimeStore();
  const videoMatches = matches.filter(m => m.has_video).length;

  // Poll /health so the badge distinguishes "API down" from "API up + no
  // analyze session running". Without this the badge shows OFFLINE whenever
  // no WebSocket is open, which for a dataset with no videos downloaded
  // looks identical to "backend is broken".
  const [apiOnline, setApiOnline] = useState(false);
  useEffect(() => {
    let mounted = true;
    const check = () => fetchHealth().then(
      () => { if (mounted) setApiOnline(true); },
      () => { if (mounted) setApiOnline(false); },
    );
    check();
    const id = setInterval(check, 5000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  const status = isConnected ? 'live' : apiOnline ? 'ready' : 'offline';
  const statusLabel = status === 'live' ? 'LIVE' : status === 'ready' ? 'READY' : 'OFFLINE';

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#4fc3f7" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            <path d="M2 12h20" />
          </svg>
          <h1>Soccer Analytics</h1>
        </div>
        <div className="header-status">
          <div className={`live-indicator ${status}`} title={
            status === 'live' ? 'WebSocket streaming an analyze session' :
            status === 'ready' ? 'API connected — click Analyze on a match with video to go LIVE' :
            'API unreachable — start the backend with: uvicorn src.api.app:app --port 8000'
          }>
            <span className={`live-dot ${status}`} />
            <span>{statusLabel}</span>
          </div>
          <span className="header-stat">{matches.length} matches</span>
          <span className="header-stat">{videoMatches} with video</span>
          {selectedMatch && <span className="header-stat active">{selectedMatch.name}</span>}
          <span className="version">v3.0</span>
        </div>
      </header>

      <div className="app-layout">
        <aside className="sidebar">
          <MatchSelector />
        </aside>

        <main className="main-content">
          <div className="grid-top">
            <VideoPlayer />
            <EventTimeline />
          </div>
          <div className="grid-bottom">
            <AnalyticsPanel />
            {isConnected ? (
              <RealtimePitchView
                players={currentFrame?.players || []}
                videoWidth={videoInfo?.width || 1280}
                videoHeight={videoInfo?.height || 720}
                isProcessing={isConnected}
                stats={stats}
              />
            ) : (
              <PitchHeatmap />
            )}
          </div>
          {isConnected && (
            <div className="grid-realtime">
              <LiveStatsPanel
                isConnected={isConnected}
                isProcessing={isConnected}
                stats={stats}
                overlaySettings={{ showBoundingBoxes, showTrackIds, showConfidence, showTrails }}
                onToggleBoundingBoxes={toggleBoundingBoxes}
                onToggleTrackIds={toggleTrackIds}
                onToggleConfidence={toggleConfidence}
                onToggleTrails={toggleTrails}
              />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
