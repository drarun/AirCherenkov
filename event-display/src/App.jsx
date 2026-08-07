import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { ChevronLeft, ChevronRight, Activity, Zap, Target } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE = 'http://localhost:8000/api';

// Simple viridis-like colormap function
function getViridisColor(value, max) {
  if (max === 0) return 'rgba(0,0,0,0)';
  let v = Math.max(0, Math.min(1, value / max));
  // 0 to 1 mapping to a color
  // simple blue-purple-orange-yellow
  const stops = [
    {v: 0.0, r: 68, g: 1, b: 84},
    {v: 0.25, r: 59, g: 82, b: 139},
    {v: 0.5, r: 33, g: 145, b: 140},
    {v: 0.75, r: 94, g: 201, b: 98},
    {v: 1.0, r: 253, g: 231, b: 37}
  ];
  
  if (v <= 0) return `rgb(${stops[0].r}, ${stops[0].g}, ${stops[0].b})`;
  if (v >= 1) return `rgb(${stops[4].r}, ${stops[4].g}, ${stops[4].b})`;
  
  for (let i = 0; i < stops.length - 1; i++) {
    if (v >= stops[i].v && v <= stops[i+1].v) {
      const t = (v - stops[i].v) / (stops[i+1].v - stops[i].v);
      const r = Math.round(stops[i].r + t * (stops[i+1].r - stops[i].r));
      const g = Math.round(stops[i].g + t * (stops[i+1].g - stops[i].g));
      const b = Math.round(stops[i].b + t * (stops[i+1].b - stops[i].b));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  return 'rgb(0,0,0)';
}

function Hexagon({ cx, cy, r, fill, onClick }) {
  const points = [];
  for (let i = 0; i < 6; i++) {
    const angle = (2 * Math.PI / 6) * i + Math.PI / 6;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    points.push(`${x},${y}`);
  }
  return (
    <polygon 
      points={points.join(' ')} 
      fill={fill} 
      className="hex-pixel"
      onClick={onClick}
    />
  );
}

function TelescopeCamera({ index, pixelX, pixelY, charges, maxCharge, onPixelClick, traces }) {
  const r = useMemo(() => {
    if (!pixelX || pixelX.length < 2) return 1;
    let minD = Infinity;
    for (let i = 1; i < pixelX.length; i++) {
      const d = Math.hypot(pixelX[i] - pixelX[0], pixelY[i] - pixelY[0]);
      if (d > 0 && d < minD) minD = d;
    }
    return minD / 1.732; // Approx radius for packing
  }, [pixelX, pixelY]);

  const viewBox = useMemo(() => {
    if (!pixelX || pixelX.length === 0) return "0 0 100 100";
    const minX = Math.min(...pixelX) - r * 1.5;
    const maxX = Math.max(...pixelX) + r * 1.5;
    const minY = Math.min(...pixelY) - r * 1.5;
    const maxY = Math.max(...pixelY) + r * 1.5;
    return `${minX} ${minY} ${maxX - minX} ${maxY - minY}`;
  }, [pixelX, pixelY, r]);

  return (
    <div className="camera-view glass">
      <div className="camera-title">
        Telescope {index + 1}
      </div>
      <div className="camera-svg-container">
        <svg viewBox={viewBox} preserveAspectRatio="xMidYMid meet" className="camera-svg">
          {pixelX.map((x, i) => {
            const charge = charges[i] || 0;
            const fill = charge > 0 ? getViridisColor(charge, maxCharge) : 'rgba(255,255,255,0.02)';
            return (
              <Hexagon 
                key={i} 
                cx={x} 
                cy={pixelY[i]} 
                r={r*0.95} 
                fill={fill} 
                onClick={() => onPixelClick(index, i)}
              />
            );
          })}
        </svg>
      </div>
    </div>
  );
}

export default function App() {
  const [config, setConfig] = useState(null);
  const [eventId, setEventId] = useState(0);
  const [eventData, setEventData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedPixel, setSelectedPixel] = useState(null);

  useEffect(() => {
    axios.get(`${API_BASE}/config`).then(res => {
      setConfig(res.data);
      fetchEvent(0);
    }).catch(err => {
      console.error("Failed to load config", err);
      setLoading(false);
    });
  }, []);

  const fetchEvent = (id) => {
    setLoading(true);
    axios.get(`${API_BASE}/events/${id}`).then(res => {
      setEventData(res.data);
      setEventId(id);
      setSelectedPixel(null);
      setLoading(false);
    }).catch(err => {
      console.error("Failed to load event", err);
      setLoading(false);
    });
  };

  if (!config) {
    return (
      <div className="app-container" style={{ justifyContent: 'center', alignItems: 'center' }}>
        <div className="loader"><Activity size={48} className="spin" /> Loading Spatiotemporal Data...</div>
      </div>
    );
  }

  const maxCharge = eventData ? Math.max(...eventData.charge.flat()) : 1;

  const handlePrev = () => {
    if (eventId > 0) fetchEvent(eventId - 1);
  };
  const handleNext = () => {
    if (eventId < config.num_events - 1) fetchEvent(eventId + 1);
  };

  const getChartData = () => {
    if (!selectedPixel || !eventData) return null;
    const { tel, pix } = selectedPixel;
    const trace = eventData.fadc_traces[tel][pix];
    return {
      labels: Array.from({length: trace.length}, (_, i) => `Bin ${i}`),
      datasets: [
        {
          label: `Tel ${tel+1} Pixel ${pix}`,
          data: trace,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.2)',
          tension: 0.3,
          fill: true,
        }
      ]
    };
  };
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: selectedPixel ? `FADC Trace: Tel ${selectedPixel.tel+1}, Pixel ${selectedPixel.pix}` : '', color: '#94a3b8' }
    },
    scales: {
      y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
      x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
    }
  };

  return (
    <div className="app-container">
      <header className="header glass">
        <h1 className="title">
          <Zap size={28} color="#38bdf8" />
          Cherenkov Event Viewer
        </h1>
        <div className="controls">
          <span style={{ color: 'var(--text-muted)' }}>
            Event {eventId + 1} / {config.num_events}
          </span>
          <button className="btn" onClick={handlePrev} disabled={eventId === 0 || loading}>
            <ChevronLeft size={18} /> Prev
          </button>
          <button className="btn btn-primary" onClick={handleNext} disabled={eventId === config.num_events - 1 || loading}>
            Next <ChevronRight size={18} />
          </button>
        </div>
      </header>

      <div className="main-content">
        <aside className="sidebar glass">
          <h2 style={{marginTop: 0, display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
            <Activity size={20} color="#818cf8"/> Event Details
          </h2>
          {loading ? (
            <div className="loader" style={{height: 100}}><Activity className="spin" size={24}/></div>
          ) : eventData ? (
            <>
              <div className="stat-box">
                <div className="stat-label">True Energy (TeV)</div>
                <div className="stat-value text-accent">{(eventData.energy / 1000).toFixed(3)}</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Particle Label</div>
                <div className="stat-value">{eventData.label === 1 ? 'Gamma' : 'Proton'}</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Impact X (m)</div>
                <div className="stat-value">{eventData.impact_x.toFixed(2)}</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Impact Y (m)</div>
                <div className="stat-value">{eventData.impact_y.toFixed(2)}</div>
              </div>

              {selectedPixel ? (
                <div className="trace-chart-container">
                  <Line data={getChartData()} options={chartOptions} />
                </div>
              ) : (
                <div className="trace-chart-container" style={{display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', color: 'var(--text-muted)'}}>
                  Click any colored pixel to view its FADC temporal trace.
                </div>
              )}
            </>
          ) : null}
        </aside>

        <main className="cameras-grid">
          {eventData && [0, 1, 2, 3].map((i) => (
            <TelescopeCamera 
              key={i}
              index={i}
              pixelX={config.pixel_x}
              pixelY={config.pixel_y}
              charges={eventData.charge[i]}
              maxCharge={maxCharge}
              onPixelClick={(tel, pix) => setSelectedPixel({tel, pix})}
            />
          ))}
        </main>
      </div>
    </div>
  );
}
