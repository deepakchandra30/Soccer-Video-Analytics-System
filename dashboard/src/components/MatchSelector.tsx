import React, { useEffect, useRef } from 'react';
import { useStore, Match } from '../store';
import { fetchMatches } from '../api';
import { Search, Trophy, Calendar, CheckCircle2, Circle } from 'lucide-react';

export const MatchSelector: React.FC = () => {
  const { matches, selectedMatch, searchQuery, setMatches, selectMatch, setSearchQuery } = useStore();
  const didInit = useRef(false);

  useEffect(() => {
    if (didInit.current) return;
    didInit.current = true;

    fetchMatches().then((data: Match[]) => {
      setMatches(data);
      if (data.length > 0) {
        const withVideo = data.find(m => m.has_video);
        selectMatch(withVideo || data[0]);
      }
    }).catch(console.error);
  }, [setMatches, selectMatch]);

  const filtered = matches.filter(m =>
    m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.league.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const grouped: Record<string, Match[]> = {};
  filtered.forEach(m => {
    const key = `${m.league} — ${m.season}`;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(m);
  });

  return (
    <div className="match-selector">
      <div className="search-box">
        <Search size={16} />
        <input
          type="text"
          placeholder="Search matches..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
        />
      </div>

      <div className="match-count">{filtered.length} matches</div>

      <div className="match-groups">
        {Object.entries(grouped).map(([group, groupMatches]) => (
          <div key={group} className="match-group">
            <div className="group-header">
              <Trophy size={14} />
              <span>{group}</span>
            </div>
            {groupMatches.map(m => (
              <div
                key={m.id}
                className={`match-item ${selectedMatch?.id === m.id ? 'active' : ''}`}
                onClick={() => selectMatch(m)}
              >
                <div className="match-name">{m.name}</div>
                <div className="match-meta">
                  <Calendar size={12} />
                  <span>{m.date}</span>
                  {m.has_features && <CheckCircle2 size={12} className="status-icon ok" />}
                  {!m.has_features && <Circle size={12} className="status-icon" />}
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};
