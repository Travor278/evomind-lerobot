'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { RobotTrajectory3DScene } from './scene';
import { buildRobotTrajectory } from './trajectory';
import type { RobotEpisode, RobotModelManifest, RobotTrajectory, RobotTrajectorySignal } from './types';

export function RobotTrajectory3DPanel({ episode, currentTime }: { episode: RobotEpisode; currentTime: number }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<RobotTrajectory3DScene | null>(null);
  const trajectoryRef = useRef<RobotTrajectory | null>(null);
  const currentTimeRef = useRef(currentTime);
  const [signal, setSignal] = useState<RobotTrajectorySignal>('action');
  const [model, setModel] = useState<RobotModelManifest | null>(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [sceneLoading, setSceneLoading] = useState(false);
  const [modelError, setModelError] = useState('');
  const [sceneError, setSceneError] = useState('');
  const [retryToken, setRetryToken] = useState(0);
  const hasAction = episode.series.some((item) => item.action.length > 0 && item.action_name);
  const hasState = episode.series.some((item) => item.state.length > 0 && item.state_name);
  const effectiveSignal = !hasAction && hasState ? 'state' : signal;
  const result = useMemo(
    () => model ? buildRobotTrajectory(episode, effectiveSignal, model) : { trajectory: null, error: '' },
    [effectiveSignal, episode, model],
  );
  const visibleSides = useMemo(() => {
    if (!result.trajectory) return ['left'] as const;
    return (['left', 'right'] as const).filter((side) => Object.keys(result.trajectory?.arms[side].joint_values ?? {}).length > 0);
  }, [result.trajectory]);

  useEffect(() => { currentTimeRef.current = currentTime; }, [currentTime]);
  useEffect(() => { trajectoryRef.current = result.trajectory; }, [result.trajectory]);

  useEffect(() => {
    let active = true;
    if (!episode.robot_type) return undefined;
    void fetch(`/api/dataset/robot-model/${encodeURIComponent(episode.robot_type)}`, { cache: 'no-store' })
      .then(async (response) => {
        if (!response.ok) {
          const payload = await response.json().catch(() => ({})) as { detail?: string };
          throw new Error(payload.detail || '3D 模型清单读取失败');
        }
        return response.json() as Promise<RobotModelManifest>;
      })
      .then((value) => { if (active) { setModel(value); setSceneLoading(true); } })
      .catch((error: unknown) => { if (active) setModelError(error instanceof Error ? error.message : '3D 模型清单读取失败'); })
      .finally(() => { if (active) setModelLoading(false); });
    return () => { active = false; };
  }, [episode.robot_type, retryToken]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !model || visibleSides.length === 0) return undefined;
    let active = true;
    const scene = new RobotTrajectory3DScene(container);
    sceneRef.current = scene;
    void scene.loadArms(model, [...visibleSides])
      .then(() => {
        if (!active) return;
        setSceneLoading(false);
        if (trajectoryRef.current) scene.applyTime(trajectoryRef.current, model, currentTimeRef.current);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setSceneLoading(false);
        setSceneError(error instanceof Error ? error.message : '3D 场景加载失败');
      });
    return () => {
      active = false;
      sceneRef.current = null;
      scene.dispose();
    };
  }, [model, visibleSides]);

  useEffect(() => {
    if (sceneRef.current && model && result.trajectory) sceneRef.current.applyTime(result.trajectory, model, currentTime);
  }, [currentTime, model, result.trajectory]);

  const error = (!episode.robot_type ? '数据集没有记录 robot_type' : '') || modelError || sceneError || result.error;
  const retry = () => {
    setModel(null);
    setModelError('');
    setSceneError('');
    setModelLoading(true);
    setRetryToken((value) => value + 1);
  };
  return <section className="robot-trajectory3d-panel">
    <div className="robot-trajectory3d-heading">
      <div><span>3D 回放</span><strong>{model?.model === 'piperx' ? 'PiperX' : model?.model === 'so101' ? 'SO-101' : episode.robot_type || '机械臂'}</strong></div>
      <div className="robot-trajectory3d-signal" role="group" aria-label="3D 轨迹信号源">
        <button type="button" className={effectiveSignal === 'action' ? 'active' : ''} disabled={!hasAction} aria-pressed={effectiveSignal === 'action'} onClick={() => setSignal('action')}>Action</button>
        <button type="button" className={effectiveSignal === 'state' ? 'active' : ''} disabled={!hasState} aria-pressed={effectiveSignal === 'state'} onClick={() => setSignal('state')}>Observation</button>
      </div>
    </div>
    <div className="robot-trajectory3d-viewport" ref={containerRef}>
      {episode.robot_type && (modelLoading || sceneLoading) && <div className="robot-trajectory3d-overlay">正在加载 3D 模型</div>}
      {error && <div className="robot-trajectory3d-overlay error"><strong>3D 回放不可用</strong><span>{error}</span><button type="button" className="outline" onClick={retry}>重试</button></div>}
    </div>
  </section>;
}
