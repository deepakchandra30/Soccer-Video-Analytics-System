import axios from 'axios';

const API = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
});

export const fetchHealth = () => API.get('/health').then(r => r.data);
export const fetchMatches = () => API.get('/matches').then(r => r.data);
export const fetchEvents = (id: string) => API.get(encodeURI(`/matches/${id}/events`)).then(r => r.data);
export const fetchAnalytics = (id: string) => API.get(encodeURI(`/matches/${id}/analytics`)).then(r => r.data);
export const fetchTimeline = (id: string) => API.get(encodeURI(`/matches/${id}/timeline`)).then(r => r.data);
export const getVideoUrl = (id: string, half: number = 1) =>
  encodeURI(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/video/${id}?half=${half}`);

export const fetchRealtimeStatus = () => API.get('/realtime/status').then(r => r.data);
export const startRealtimeSession = (id: string) => API.post(encodeURI(`/realtime/start/${id}`)).then(r => r.data);
export const stopRealtimeSession = (id: string) => API.post(encodeURI(`/realtime/stop/${id}`)).then(r => r.data);
export const fetchRealtimeStats = (id: string) => API.get(encodeURI(`/realtime/stats/${id}`)).then(r => r.data);
export const fetchRealtimeFrame = (id: string, timestampMs: number) =>
  API.get(encodeURI(`/realtime/frame/${id}`), { params: { timestamp_ms: timestampMs } }).then(r => r.data);
