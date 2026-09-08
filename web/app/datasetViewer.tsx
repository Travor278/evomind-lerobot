'use client';

import { Pause, Play } from 'lucide-react';
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';

const RobotTrajectory3DPanel = lazy(() => import('./robotTrajectory3D/RobotTrajectory3DPanel').then((module) => ({ default: module.RobotTrajectory3DPanel })));

type RuntimeEvent = {
  sequence: number; operation: string; phase: string; message: string;
  data: Record<string, unknown>; timestamp: string;
};
type WorkflowRuntime = { running: boolean; operation: string | null; event: RuntimeEvent | null };
type DatasetSummary = {
  id: string; path: string; episodes: number; frames: number; fps: number; duration_s: number;
  tasks: string[]; camera_count: number; status: 'ready' | 'recording' | 'incomplete' | 'unreadable';
  robot_type: string; recorded_on: string; available: boolean; error: string;
};
type DatasetDetail = {
  id: string; robot_type: string | null; fps: number; frames: number; duration_s: number; tasks: string[];
  cameras: { key: string; label: string; resolution: string | null; depth: boolean }[];
  episodes: { episode_index: number; frames: number; duration_s: number; tasks: string[] }[];
};
type EpisodeSeries = { label: string; action_name: string; state_name: string; action: number[]; state: number[] };
type EpisodePayload = {
  dataset_id: string; robot_type: string | null; episode_index: number; frames: number; duration_s: number; fps: number; tasks: string[];
  timestamps: number[]; series: EpisodeSeries[];
  videos: { key: string; label: string; url: string; from_timestamp: number; to_timestamp: number }[];
};

const COLORS = ['#111111', '#2563eb', '#dc2626', '#16a34a', '#9333ea', '#d97706', '#0891b2', '#db2777', '#4f46e5', '#65a30d', '#7c3aed', '#ea580c'];

async function read<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(payload.detail || '数据读取失败');
  }
  return response.json() as Promise<T>;
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(payload.detail || '操作失败');
  }
  return response.json() as Promise<T>;
}

function durationLabel(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return minutes ? `${minutes}分 ${remainder}秒` : `${remainder}秒`;
}

function countLabel(value: number) {
  return new Intl.NumberFormat('zh-CN').format(value);
}

export function DatasetViewerPage({ runtimeEvent, robotType }: { runtimeEvent: RuntimeEvent | null; robotType: string; statusSlot: HTMLDivElement | null }) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [episodeIndex, setEpisodeIndex] = useState(0);
  const [episode, setEpisode] = useState<EpisodePayload | null>(null);
  const [query, setQuery] = useState('');
  const [taskFilter, setTaskFilter] = useState('');
  const [dateFilter, setDateFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const [episodeLoading, setEpisodeLoading] = useState(false);
  const [error, setError] = useState('');
  const [runtime, setRuntime] = useState<WorkflowRuntime>({ running: false, operation: null, event: null });
  const [replayPending, setReplayPending] = useState(false);

  const refreshDatasets = useCallback(async () => {
    try {
      const values = await read<DatasetSummary[]>('/api/datasets');
      const visible = values.filter((item) => item.status === 'unreadable' || (item.episodes > 0 && item.frames > 0 && item.duration_s > 0));
      setDatasets(visible); setError('');
      setSelectedId((current) => visible.some((item) => item.id === current) ? current : '');
    } catch (refreshError) { setError(refreshError instanceof Error ? refreshError.message : '数据集读取失败'); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void refreshDatasets(), 0);
    return () => window.clearTimeout(timer);
  }, [refreshDatasets]);
  useEffect(() => {
    const refresh = () => read<WorkflowRuntime>('/api/runtime/status').then(setRuntime).catch(() => undefined);
    void refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    if (runtimeEvent?.operation !== 'recording' || !['completed', 'failed'].includes(runtimeEvent.phase)) return undefined;
    const timer = window.setTimeout(() => void refreshDatasets(), 0);
    return () => window.clearTimeout(timer);
  }, [refreshDatasets, runtimeEvent]);

  useEffect(() => {
    if (!selectedId) return undefined;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setEpisodeIndex(0); setEpisode(null); setError('');
      void read<DatasetDetail>(`/api/dataset/detail?dataset_id=${encodeURIComponent(selectedId)}`)
        .then((value) => { if (!cancelled) setDetail(value); })
        .catch((detailError) => { if (!cancelled) setError(detailError instanceof Error ? detailError.message : '数据集详情读取失败'); });
    }, 0);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId) return undefined;
    const close = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSelectedId('');
    };
    document.addEventListener('keydown', close);
    return () => document.removeEventListener('keydown', close);
  }, [selectedId]);

  useEffect(() => {
    if (!detail || !detail.episodes.some((item) => item.episode_index === episodeIndex)) return undefined;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setEpisodeLoading(true); setError('');
      void read<EpisodePayload>(`/api/dataset/episode?dataset_id=${encodeURIComponent(detail.id)}&episode=${episodeIndex}`)
        .then((value) => { if (!cancelled) setEpisode(value); })
        .catch((episodeError) => { if (!cancelled) setError(episodeError instanceof Error ? episodeError.message : 'Episode 读取失败'); })
        .finally(() => { if (!cancelled) setEpisodeLoading(false); });
    }, 0);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [detail, episodeIndex]);

  const taskOptions = Array.from(new Set(datasets.flatMap((dataset) => dataset.tasks))).sort((left, right) => left.localeCompare(right, 'zh-CN'));
  const filtered = datasets.filter((dataset) => {
    const text = `${dataset.id} ${dataset.tasks.join(' ')}`.toLowerCase();
    return text.includes(query.trim().toLowerCase())
      && (!taskFilter || dataset.tasks.includes(taskFilter))
      && (!dateFilter || dataset.recorded_on === dateFilter);
  });

  const replayEvent = runtimeEvent?.operation === 'replay' ? runtimeEvent : runtime.event?.operation === 'replay' ? runtime.event : null;
  const replaying = runtime.running && runtime.operation === 'replay';
  const runningOther = runtime.running && runtime.operation !== 'replay';
  const activeReplayId = String(replayEvent?.data.repo_id ?? '');

  async function startReplay(datasetId: string, selectedEpisode = 0) {
    setReplayPending(true); setError('');
    try { setRuntime(await post<WorkflowRuntime>('/api/runtime/replay/start', { dataset_id: datasetId, episode: selectedEpisode })); }
    catch (replayError) { setError(replayError instanceof Error ? replayError.message : '回放启动失败'); }
    finally { setReplayPending(false); }
  }

  async function stopReplay() {
    setReplayPending(true); setError('');
    try { setRuntime(await post<WorkflowRuntime>('/api/runtime/command', { command: 'stop' })); }
    catch (replayError) { setError(replayError instanceof Error ? replayError.message : '回放停止失败'); }
    finally { setReplayPending(false); }
  }

  return <section className="dataset-page">
    <section className="dataset-filter-bar">
      <div className="dataset-filter-heading"><div><strong>筛选条件</strong><span>按数据集、任务和采集日期查找</span></div>{(query || taskFilter || dateFilter) && <button type="button" onClick={() => { setQuery(''); setTaskFilter(''); setDateFilter(''); }}>清除筛选</button>}</div>
      <div className="dataset-filter-fields">
        <label>数据集<input value={query} onChange={(item) => setQuery(item.target.value)} placeholder="搜索数据集名称" aria-label="搜索数据集" /></label>
        <label>任务<select value={taskFilter} onChange={(item) => setTaskFilter(item.target.value)}><option value="">全部任务</option>{taskOptions.map((task) => <option value={task} key={task}>{task}</option>)}</select></label>
        <label>采集日期<input type="date" value={dateFilter} onChange={(item) => setDateFilter(item.target.value)} /></label>
      </div>
    </section>
    {error && <div className="error compact">{error}</div>}
    <section className="dataset-management-board">
      <div className="dataset-management-heading"><h2>我的数据</h2><span>{filtered.length}</span></div>
      <div className="dataset-management-table">
        <div className="dataset-management-table-head"><span>数据集</span><span>Episodes</span><span>有效时长</span><span>总帧数</span><span>FPS / 相机</span><span>操作</span></div>
        {filtered.map((dataset) => {
          const compatible = Boolean(dataset.robot_type) && dataset.robot_type === robotType;
          const activeReplay = replaying && activeReplayId === dataset.id;
          const replayDisabled = replayPending || runningOther || (replaying && !activeReplay) || !compatible || !dataset.available;
          const openDataset = () => { if (!dataset.available) return; setDetail(null); setEpisode(null); setSelectedId(dataset.id); };
          const unavailable = !dataset.available;
          return <div className={`dataset-management-row${unavailable ? ' unavailable' : ''}`} role={unavailable ? undefined : 'button'} aria-disabled={unavailable || undefined} tabIndex={unavailable ? -1 : 0} onClick={openDataset} onKeyDown={(event) => { if (!unavailable && event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); openDataset(); } }} key={dataset.id}>
            <span><strong>{dataset.id}</strong><small>{dataset.status === 'unreadable' ? '不可读取' : `${dataset.tasks[0] || '未记录任务'} · ${dataset.recorded_on || '日期未知'}`}</small></span>
            <span>{countLabel(dataset.episodes)}</span>
            <span>{durationLabel(dataset.duration_s)}</span>
            <span>{countLabel(dataset.frames)}</span>
            <span>{dataset.fps || '—'} / {dataset.camera_count}</span>
            <span className="dataset-row-actions" onClick={(event) => event.stopPropagation()}>
              <button className="outline" type="button" disabled={unavailable} title={dataset.error || undefined} onClick={openDataset}>{dataset.status === 'unreadable' ? '不可读取' : '可视化'}</button>
              <button className={activeReplay ? 'danger' : 'outline'} type="button" disabled={replayDisabled} title={compatible ? undefined : `仅支持 ${robotType} 数据`} onClick={() => activeReplay ? void stopReplay() : void startReplay(dataset.id)}>{activeReplay ? '停止' : compatible ? '回放' : '不兼容'}</button>
            </span>
            {dataset.error && <em>{dataset.error}</em>}
          </div>;
        })}
        {!loading && filtered.length === 0 && <div className="empty-state">没有匹配的本地数据集</div>}
        {loading && <div className="empty-state">正在扫描本地数据</div>}
      </div>
    </section>

    {selectedId && <>
      <div className="dataset-drawer-backdrop" role="presentation" onClick={() => setSelectedId('')} />
      <aside className="dataset-drawer" role="dialog" aria-modal="true" aria-label={`${selectedId} 数据可视化`}>
        <div className="dataset-drawer-header"><div><span>数据可视化</span><h2>{selectedId}</h2></div><button type="button" onClick={() => setSelectedId('')} aria-label="关闭数据可视化">×</button></div>
        <div className="dataset-drawer-content">
          {!detail && <div className="dataset-placeholder">正在加载数据集</div>}
          {detail && <>
            <section className="dataset-overview">
              <div><span>任务</span><p>{detail.tasks.join(' · ') || '未记录任务描述'}</p></div>
              <div className="dataset-overview-metrics">
                <Metric label="Episodes" value={countLabel(detail.episodes.length)} />
                <Metric label="总帧数" value={countLabel(detail.frames)} />
                <Metric label="有效时长" value={durationLabel(detail.duration_s)} />
                <Metric label="FPS / 相机" value={`${detail.fps} / ${detail.cameras.length}`} />
              </div>
            </section>

            <section className="episode-toolbar">
              <label>Episode<select value={episodeIndex} onChange={(item) => setEpisodeIndex(Number(item.target.value))}>{detail.episodes.map((item) => <option value={item.episode_index} key={item.episode_index}>Episode {item.episode_index} · {durationLabel(item.duration_s)}</option>)}</select></label>
              <div className="episode-toolbar-meta">{detail.cameras.map((camera) => <span key={camera.key}>{camera.label}{camera.resolution ? ` · ${camera.resolution}` : ''}</span>)}</div>
              <div className="episode-toolbar-actions"><button className={replaying && activeReplayId === detail.id ? 'danger' : 'primary'} type="button" disabled={replayPending || runningOther || (replaying && activeReplayId !== detail.id) || detail.robot_type !== robotType} onClick={() => replaying && activeReplayId === detail.id ? void stopReplay() : void startReplay(detail.id, episodeIndex)}>{replaying && activeReplayId === detail.id ? '停止回放' : detail.robot_type === robotType ? '回放此 Episode' : '设备类型不兼容'}</button></div>
            </section>

            {episodeLoading && <div className="dataset-placeholder compact">正在加载 Episode</div>}
            {episode && !episodeLoading && <EpisodePlayer key={`${episode.dataset_id}-${episode.episode_index}`} episode={episode} />}
          </>}
        </div>
      </aside>
    </>}
  </section>;
}

function EpisodePlayer({ episode }: { episode: EpisodePayload }) {
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [playbackError, setPlaybackError] = useState('');
  const videoRefs = useRef<Array<HTMLVideoElement | null>>([]);
  const animationRef = useRef<number | null>(null);
  const currentRef = useRef(0);

  const seek = useCallback((seconds: number, force = true) => {
    const bounded = Math.max(0, Math.min(episode.duration_s, seconds));
    currentRef.current = bounded; setCurrentTime(bounded);
    videoRefs.current.forEach((video, index) => {
      if (!video || video.readyState === 0) return;
      const target = episode.videos[index].from_timestamp + bounded;
      if (force || Math.abs(video.currentTime - target) > 0.15) video.currentTime = target;
    });
  }, [episode]);

  useEffect(() => {
    videoRefs.current.forEach((video) => { if (video) video.playbackRate = rate; });
  }, [rate]);

  useEffect(() => {
    const videos = videoRefs.current.filter((item): item is HTMLVideoElement => Boolean(item));
    if (!playing) { videos.forEach((video) => video.pause()); return undefined; }
    if (currentRef.current >= episode.duration_s - 0.01) seek(0);
    videos.forEach((video) => {
      const index = videoRefs.current.indexOf(video);
      video.currentTime = episode.videos[index].from_timestamp + currentRef.current;
      video.playbackRate = rate;
      void video.play().catch((reason: unknown) => {
        setPlaybackError(reason instanceof Error ? reason.message : '视频播放失败'); setPlaying(false);
      });
    });
    const tick = () => {
      const leader = videoRefs.current[0];
      if (leader) {
        const relative = leader.currentTime - episode.videos[0].from_timestamp;
        if (relative >= episode.duration_s - 0.02) { seek(episode.duration_s); setPlaying(false); return; }
        currentRef.current = Math.max(0, relative); setCurrentTime(Math.max(0, relative));
        videoRefs.current.slice(1).forEach((video, index) => {
          if (!video) return;
          const target = episode.videos[index + 1].from_timestamp + relative;
          if (Math.abs(video.currentTime - target) > 0.15) video.currentTime = target;
        });
      }
      animationRef.current = window.requestAnimationFrame(tick);
    };
    animationRef.current = window.requestAnimationFrame(tick);
    return () => { if (animationRef.current !== null) window.cancelAnimationFrame(animationRef.current); };
  }, [episode, playing, rate, seek]);

  return <div className="episode-player">
    {episode.tasks[0] && <div className="episode-task"><span>任务描述</span><strong>{episode.tasks.join(' · ')}</strong></div>}
    {playbackError && <div className="error compact">{playbackError}</div>}
    <div className="camera-grid">
      {episode.videos.map((video, index) => <figure key={video.key}><video ref={(element) => { videoRefs.current[index] = element; }} src={video.url} preload="metadata" playsInline onLoadedMetadata={(event) => { event.currentTarget.currentTime = video.from_timestamp; }} /><figcaption><strong>{video.label}</strong><span>{durationLabel(episode.duration_s)}</span></figcaption></figure>)}
      {episode.videos.length === 0 && <div className="empty-state">这个 Episode 没有视频流</div>}
    </div>
    <div className="playback-controls">
      <button className="play-button" type="button" onClick={() => setPlaying((value) => !value)} aria-label={playing ? '暂停' : '播放'}>{playing ? <Pause size={17} /> : <Play size={17} />}</button>
      <span>{durationLabel(currentTime)}</span>
      <input type="range" min="0" max={Math.max(episode.duration_s, 0.01)} step={1 / Math.max(episode.fps, 1)} value={currentTime} onChange={(item) => seek(Number(item.target.value))} aria-label="Episode 时间轴" />
      <span>{durationLabel(episode.duration_s)}</span>
      <select value={rate} onChange={(item) => setRate(Number(item.target.value))} aria-label="播放速度"><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select>
    </div>
    <Suspense fallback={<div className="dataset-placeholder compact">正在加载 3D 回放</div>}><RobotTrajectory3DPanel episode={episode} currentTime={currentTime} /></Suspense>
    <TrajectoryChart episode={episode} currentTime={currentTime} hidden={hidden} onHiddenChange={setHidden} onSeek={seek} />
  </div>;
}

function TrajectoryChart({ episode, currentTime, hidden, onHiddenChange, onSeek }: { episode: EpisodePayload; currentTime: number; hidden: Set<number>; onHiddenChange: (value: Set<number>) => void; onSeek: (seconds: number) => void }) {
  const visible = episode.series.filter((_, index) => !hidden.has(index));
  const values = visible.flatMap((series) => [...series.action, ...series.state]).filter(Number.isFinite);
  const rawMin = values.reduce((minimum, value) => Math.min(minimum, value), Number.POSITIVE_INFINITY);
  const rawMax = values.reduce((maximum, value) => Math.max(maximum, value), Number.NEGATIVE_INFINITY);
  const boundedMin = Number.isFinite(rawMin) ? rawMin : -1;
  const boundedMax = Number.isFinite(rawMax) ? rawMax : 1;
  const padding = Math.max((boundedMax - boundedMin) * 0.08, 0.01);
  const minimum = boundedMin - padding;
  const maximum = boundedMax + padding;
  const cursorX = 6 + Math.max(0, Math.min(1, currentTime / Math.max(episode.duration_s, 0.001))) * 92;

  function toggle(index: number) {
    const next = new Set(hidden);
    if (next.has(index)) next.delete(index); else next.add(index);
    onHiddenChange(next);
  }

  if (!episode.series.length) return <div className="empty-state">这个 Episode 没有 action / observation.state 曲线</div>;
  return <section className="trajectory-panel">
    <div className="trajectory-heading"><div><span>轨迹</span><strong>Action / Observation State</strong></div><span>{countLabel(episode.frames)} 帧</span></div>
    <button className="trajectory-plot" type="button" onClick={(event) => { const bounds = event.currentTarget.getBoundingClientRect(); onSeek((event.clientX - bounds.left) / bounds.width * episode.duration_s); }} aria-label="点击曲线跳转时间">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <g className="trajectory-grid">{[12, 29, 46, 63, 80].map((y) => <line key={y} x1="6" y1={y} x2="98" y2={y} />)}{[6, 24.4, 42.8, 61.2, 79.6, 98].map((x) => <line key={x} x1={x} y1="12" x2={x} y2="80" />)}</g>
        {episode.series.map((series, index) => hidden.has(index) ? null : <g key={`${series.label}-${index}`}>
          {series.action.length > 0 && <polyline points={polyline(series.action, episode.timestamps, episode.duration_s, minimum, maximum)} fill="none" stroke={COLORS[index % COLORS.length]} strokeWidth="0.45" vectorEffect="non-scaling-stroke" />}
          {series.state.length > 0 && <polyline points={polyline(series.state, episode.timestamps, episode.duration_s, minimum, maximum)} fill="none" stroke={COLORS[index % COLORS.length]} strokeWidth="0.45" strokeDasharray="3 2" vectorEffect="non-scaling-stroke" />}
        </g>)}
        <line className="trajectory-cursor" x1={cursorX} y1="12" x2={cursorX} y2="80" vectorEffect="non-scaling-stroke" />
      </svg>
    </button>
    <div className="trajectory-legend">{episode.series.map((series, index) => <label className={hidden.has(index) ? 'hidden' : ''} key={`${series.label}-${index}`}><input type="checkbox" checked={!hidden.has(index)} onChange={() => toggle(index)} style={{ accentColor: COLORS[index % COLORS.length] }} /><i style={{ background: COLORS[index % COLORS.length] }} /><strong>{series.label}</strong><span>实线 action · 虚线 state</span></label>)}</div>
  </section>;
}

function polyline(values: number[], timestamps: number[], duration: number, minimum: number, maximum: number) {
  const range = maximum - minimum || 1;
  return values.map((value, index) => {
    const time = timestamps[index] ?? duration * index / Math.max(values.length - 1, 1);
    const x = 6 + Math.max(0, Math.min(1, time / Math.max(duration, 0.001))) * 92;
    const y = 12 + (maximum - value) / range * 68;
    return `${x.toFixed(2)},${Math.max(12, Math.min(80, y)).toFixed(2)}`;
  }).join(' ');
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}
