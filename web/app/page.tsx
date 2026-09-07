/* eslint-disable @next/next/no-img-element */
'use client';

import { Check, ChevronDown, PanelLeftClose, PanelLeftOpen, Plus, RefreshCw, Trash2, Volume2, VolumeX } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { CollectionProgressPage, StorageNotice, type StorageInfo } from './collectionProgress';
import { DatasetViewerPage } from './datasetViewer';
import {
  armModeLabel, cameraDisplayLabel, cameraKindLabel, cameraSlotId, canAddCameraKind,
  categories, createSteps, defaultCameras, modelsFromProfiles, nextCameraDraft,
  type ArmMode, type CameraDraft, type CameraKind, type CreateStepId,
  type DeviceCategory, type DeviceCategoryId, type DeviceModelOption, type DeviceTransport,
  type SystemProfile,
} from './deviceCatalog';
import { useRuntimeSounds } from './runtimeSounds';

export type RuntimeEvent = {
  sequence: number; operation: string; phase: string; message: string;
  job_id: string | null; data: Record<string, unknown>; timestamp: string;
};
type RuntimeStatus = { lerobot_version: string | null; runtime: { hostname: string } & StorageInfo; event: RuntimeEvent };
type PolicyResidency = {
  state: 'empty' | 'loading' | 'ready'; policy_path?: string; policy_type?: string; device?: string; allocated_bytes?: number | null;
};
type WorkflowRuntime = { running: boolean; job_id: string | null; operation: string | null; event: RuntimeEvent | null; policy_residency?: PolicyResidency };
type Catalog = { systems: SystemProfile[] };
type SerialDevice = { id: string; path: string; device: string };
type CanDevice = { id: string; serial_number: string; interface: string; state: string; up: boolean; bitrate: number | null };
type CameraDriver = 'opencv' | 'intelrealsense';
type CameraDevice = {
  id: string; name: string; path: string; paths: string[];
  driver: CameraDriver; serial_number: string;
};
type CameraPreview = CameraDevice & { preview_data_url: string | null; preview_error: string | null };
type HardwareInventory = { serial: SerialDevice[]; socketcan: CanDevice[]; cameras: CameraDevice[] };
type Side = 'single' | 'left' | 'right';
type SerialKind = 'robot' | 'teleoperator';
type IdentificationScope = 'hardware' | 'sensors' | 'all';
type PageId = 'device' | 'maintenance' | 'calibration' | 'teleoperation' | 'recording' | 'collection-progress' | 'datasets' | 'inference';
type SerialBinding = {
  id: string; port: string; alias: string; kind: SerialKind; side: Side;
};
type CanBinding = { id: string; alias: string; kind: SerialKind; side: Side };
type CameraBinding = {
  id: string; port: string; alias: string; side: Side;
  driver: CameraDriver; serial_number: string | null;
};
type CameraSlot = { alias: string; kind: CameraKind; side: Side };
type DeviceConfiguration = {
  profile_id: string; robot_type: string; teleoperator_type: string | null;
  camera_slots: CameraSlot[]; serial_bindings: SerialBinding[]; can_bindings: CanBinding[]; camera_bindings: CameraBinding[];
};
type HardwareSlot = { id: string; label: string; kind: SerialKind; side: Side; transport: DeviceTransport };
type MotionPort = { stable_id: string; device: string; motor_ids: number[]; delta: number; moved: boolean; motion_error: string };
type MotionStartResult = { status: string; readable_count: number; ports: MotionPort[] };
type FeetechMotor = { id: number; model: string; model_number: number; position: number; velocity: number; load: number; voltage: number; temperature: number; current: number | null; torque_enabled: boolean; moving: boolean };
type FeetechScan = { device_id: string; baudrate: number; motors: FeetechMotor[] };
type FeetechSnapshot = { positions: { id: number; position: number }[] };
type FeetechIdChange = { status: string; rescan_required: boolean; old_id: number; new_id: number; model: string };
type PositionSample = { capturedAt: number; positions: Record<number, number> };
type PiperMotor = {
  id: number; position: number | null; voltage: number; driver_temperature: number;
  motor_temperature: number; current: number; enabled: boolean; faults: string[];
};
type PiperSnapshot = {
  device_id: string; interface: string; firmware: string; feedback_source: 'feedback' | 'control' | 'none'; can_fps: number;
  status: { available: boolean; ctrl_mode: number; arm_status: number; mode: number; error_code: number };
  motors: PiperMotor[]; gripper: { available: boolean; position: number; effort: number };
};
type CalibrationStatus = {
  state: 'idle' | 'starting' | 'running' | 'stopping' | 'stopped' | 'done' | 'error';
  mode: 'auto' | 'manual' | ''; phase: string; alias: string; message: string; prompt_id: string;
  calibration_id: string; path: string; motor: string;
  profile: Record<string, unknown>; error: string; updated_at: number;
  devices: Record<string, { available: boolean; path: string; run: Omit<CalibrationStatus, 'devices'> | null }>;
};
type LocalDataset = { id: string; path: string; episodes: number; frames: number; fps: number; tasks: number };
type LocalPolicy = { id: string; path: string; type: string };
type WorkspaceInventory = { datasets: LocalDataset[]; policies: LocalPolicy[] };
type DailyCollectionTask = {
  id: string; name: string; description: string; target_duration_s: number;
  num_episodes: number; episode_time_s: number; reset_time_s: number; fps: number;
  collection_method: 'manual' | 'policy'; policy_path: string;
  rollout_strategy: Exclude<RolloutStrategy, 'base'>; inference: RolloutInference;
  duration_s: number; ring_buffer_seconds: number;
  actual_duration_s: number; episode_count: number; completed: boolean;
};
type RolloutStrategy = 'base' | 'episodic' | 'sentry' | 'highlight' | 'dagger_corrections' | 'dagger_continuous' | 'episodic_dagger';
type RolloutInference = 'sync' | 'rtc';
type PolicyInspection = {
  policy_path: string; policy_type: string; revision: string | null; size_bytes: number | null;
  state_dim: number | null; action_dim: number | null; hardware_state_dim: number | null; hardware_action_dim: number | null;
  expected_visuals: string[]; provided_visuals: string[]; rename_map: Record<string, string>;
  supports_rtc: boolean; compatible: boolean; issues: string[];
};

const rolloutModes: Record<RolloutStrategy, { label: string; detail: string }> = {
  base: { label: '纯推理', detail: 'Policy 自主控制，不记录数据' },
  episodic: { label: '分轮评测', detail: '按 Episode 记录，轮次之间可人工重置场景' },
  sentry: { label: '连续记录', detail: '持续自主推理并自动切分保存数据' },
  highlight: { label: '精彩片段', detail: '保留滚动缓存，按需保存前后片段' },
  dagger_corrections: { label: '人工纠正', detail: '自主运行，仅把人工接管窗口记录为 Episode' },
  dagger_continuous: { label: '全程 DAgger', detail: '同时记录自主与人工帧，并标记 intervention' },
  episodic_dagger: { label: 'Episodic DAgger', detail: '按 Episode 保存完整轨迹，轮内可切换 Policy 与人工干预' },
};

const menu: { id: PageId; label: string }[] = [
  { id: 'device', label: '设备' },
  { id: 'maintenance', label: '维修' },
  { id: 'calibration', label: '校准' },
  { id: 'teleoperation', label: '遥操作' },
  { id: 'inference', label: '推理' },
  { id: 'recording', label: '数据采集' },
  { id: 'collection-progress', label: '采集进度' },
  { id: 'datasets', label: '数据管理' },
];

function serialIdentity(stableId: string) {
  const match = stableId.match(/USB_Single_Serial_(.+?)-if\d+$/);
  return match?.[1] ?? stableId;
}

function serialLabel(device: Pick<SerialDevice, 'id' | 'device'>) {
  return `${device.device} · ${serialIdentity(device.id)}`;
}

async function readJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error('读取失败');
  return response.json() as Promise<T>;
}

async function requestJson<T>(url: string, method: 'POST' | 'PUT', body: unknown): Promise<T> {
  const response = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!response.ok) {
    const payload = await response.json() as { detail?: string };
    throw new Error(payload.detail ?? '操作失败');
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(url: string, body: unknown = {}): Promise<T> {
  return requestJson<T>(url, 'POST', body);
}

function newestRuntimeEvent(...events: Array<RuntimeEvent | null | undefined>): RuntimeEvent | null {
  return events.reduce<RuntimeEvent | null>((newest, candidate) => {
    if (!candidate) return newest;
    if (!newest) return candidate;
    const newestTimestamp = Date.parse(newest.timestamp) || 0;
    const candidateTimestamp = Date.parse(candidate.timestamp) || 0;
    if (candidateTimestamp !== newestTimestamp) return candidateTimestamp > newestTimestamp ? candidate : newest;
    return candidate.sequence > newest.sequence ? candidate : newest;
  }, null);
}

function rolloutPhaseLabel(phase: unknown) {
  return {
    autonomous: 'Policy 控制中',
    paused: 'Policy 已暂停',
    correcting: '人工干预中',
    resetting: '场景重置中',
  }[typeof phase === 'string' ? phase : ''];
}

function hardwareSlots(profile: SystemProfile | undefined): HardwareSlot[] {
  if (!profile) return [];
  const slot = (id: string, label: string, kind: SerialKind, side: Side): HardwareSlot => ({
    id, label, kind, side,
    transport: (kind === 'robot' ? profile.robot_transport : profile.teleoperator_transport) ?? 'serial',
  });
  if (profile.robot_ports === 2 && profile.teleoperator_ports === 2) {
    return [
      slot('left_leader_arm', '左主臂', 'teleoperator', 'left'),
      slot('left_follower_arm', '左从臂', 'robot', 'left'),
      slot('right_leader_arm', '右主臂', 'teleoperator', 'right'),
      slot('right_follower_arm', '右从臂', 'robot', 'right'),
    ];
  }
  if (profile.robot_ports === 1 && profile.teleoperator_ports === 1) {
    return [
      slot('leader_arm', '主臂', 'teleoperator', 'single'),
      slot('follower_arm', '从臂', 'robot', 'single'),
    ];
  }
  const slots: HardwareSlot[] = [];
  for (let index = 0; index < (profile.teleoperator_ports ?? 0); index += 1) slots.push(slot(`teleoperator_${index + 1}`, `遥操作设备 ${index + 1}`, 'teleoperator', 'single'));
  for (let index = 0; index < (profile.robot_ports ?? 0); index += 1) slots.push(slot(`robot_${index + 1}`, `机器人接口 ${index + 1}`, 'robot', 'single'));
  return slots;
}

function cameraDraftsFromConfiguration(configuration: DeviceConfiguration): CameraDraft[] {
  return configuration.camera_slots.map((slot, index) => ({ id: `saved-camera-${index}-${slot.alias}`, kind: slot.kind, side: slot.side === 'left' || slot.side === 'right' ? slot.side : undefined }));
}

function isReady(configuration: DeviceConfiguration | null, profile: SystemProfile | undefined) {
  if (!configuration || !profile) return false;
  const slots = hardwareSlots(profile);
  const requiredSerial = slots.filter((slot) => slot.transport === 'serial').length;
  const requiredCan = slots.filter((slot) => slot.transport === 'socketcan').length;
  return configuration.serial_bindings.length === requiredSerial
    && configuration.can_bindings.length === requiredCan
    && configuration.camera_bindings.length === configuration.camera_slots.length;
}

export default function Home() {
  const [collapsed, setCollapsed] = useState(false);
  const [activePage, setActivePage] = useState<PageId>('device');
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [catalog, setCatalog] = useState<Catalog>({ systems: [] });
  const [saved, setSaved] = useState<DeviceConfiguration | null>(null);
  const [workspace, setWorkspace] = useState<WorkspaceInventory>({ datasets: [], policies: [] });
  const [runtimeEvent, setRuntimeEvent] = useState<RuntimeEvent | null>(null);
  const [workflowStatusSlot, setWorkflowStatusSlot] = useState<HTMLDivElement | null>(null);
  const [editing, setEditing] = useState(false);
  const [step, setStep] = useState<CreateStepId>('category');
  const [categoryId, setCategoryId] = useState<DeviceCategoryId | null>(null);
  const [modelId, setModelId] = useState<string | null>(null);
  const [mode, setMode] = useState<ArmMode>('single');
  const [cameras, setCameras] = useState<CameraDraft[]>([]);
  const [pendingCameraKind, setPendingCameraKind] = useState<CameraKind>('environment');
  const [serialAssignments, setSerialAssignments] = useState<Record<string, string>>({});
  const [canAssignments, setCanAssignments] = useState<Record<string, string>>({});
  const [cameraAssignments, setCameraAssignments] = useState<Record<string, string>>({});
  const [inventory, setInventory] = useState<HardwareInventory | null>(null);
  const [cameraPreviews, setCameraPreviews] = useState<CameraPreview[]>([]);
  const [motionPorts, setMotionPorts] = useState<MotionPort[]>([]);
  const [motionActive, setMotionActive] = useState(false);
  const [motionStarting, setMotionStarting] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [identifying, setIdentifying] = useState(false);
  const [identificationScope, setIdentificationScope] = useState<IdentificationScope>('all');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const models = useMemo(() => modelsFromProfiles(catalog.systems), [catalog.systems]);
  const categoryModels = models.filter((model) => model.category === categoryId);
  const selectedCategory = categories.find((category) => category.id === categoryId);
  const selectedModel = models.find((model) => model.id === modelId);
  const selectedVariant = selectedModel?.variants.find((variant) => variant.mode === mode) ?? selectedModel?.variants[0];
  const selectedProfile = selectedVariant?.profile;
  const savedModel = models.find((model) => model.variants.some((variant) => variant.profile.id === saved?.profile_id));
  const savedProfile = savedModel?.variants.find((variant) => variant.profile.id === saved?.profile_id)?.profile;
  const ready = isReady(saved, savedProfile);
  const runtimeSounds = useRuntimeSounds(runtimeEvent);

  useEffect(() => {
    if (status?.runtime.hostname) document.title = `${status.runtime.hostname} · LeRobot`;
  }, [status?.runtime.hostname]);

  useEffect(() => {
    Promise.all([readJson<RuntimeStatus>('/api/status'), readJson<Catalog>('/api/catalog'), readJson<DeviceConfiguration | null>('/api/config'), readJson<WorkspaceInventory>('/api/workspace')])
      .then(([nextStatus, nextCatalog, configuration, nextWorkspace]) => {
        setStatus(nextStatus); setRuntimeEvent(nextStatus.event); setCatalog(nextCatalog); setSaved(configuration); setWorkspace(nextWorkspace); setEditing(!configuration);
        if (!configuration) return;
        const hydratedModel = modelsFromProfiles(nextCatalog.systems).find((model) => model.variants.some((variant) => variant.profile.id === configuration.profile_id));
        if (!hydratedModel) return;
        const variant = hydratedModel.variants.find((item) => item.profile.id === configuration.profile_id) ?? hydratedModel.variants[0];
        setCategoryId(hydratedModel.category); setModelId(hydratedModel.id); setMode(variant.mode);
        setCameras(cameraDraftsFromConfiguration(configuration));
        setSerialAssignments(Object.fromEntries(configuration.serial_bindings.map((binding) => [binding.alias, binding.id])));
        setCanAssignments(Object.fromEntries(configuration.can_bindings.map((binding) => [binding.alias, binding.id])));
      })
      .catch(() => setError('无法连接本机运行时'));
  }, []);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let disposed = false;

    const refreshLatest = () => {
      readJson<RuntimeStatus>('/api/status')
        .then((nextStatus) => setRuntimeEvent((current) => newestRuntimeEvent(current, nextStatus.event)))
        .catch(() => undefined);
    };
    const connect = () => {
      if (disposed) return;
      socket = new WebSocket(`${protocol}//${window.location.host}/api/events`);
      socket.onmessage = (message) => {
        const nextEvent = JSON.parse(message.data) as RuntimeEvent;
        setRuntimeEvent((current) => newestRuntimeEvent(current, nextEvent));
      };
      socket.onclose = () => {
        if (!disposed) reconnectTimer = window.setTimeout(connect, 1000);
      };
      socket.onerror = () => socket?.close();
    };
    const handleVisibility = () => {
      if (document.visibilityState === 'visible') refreshLatest();
    };

    connect();
    document.addEventListener('visibilitychange', handleVisibility);
    return () => {
      disposed = true;
      document.removeEventListener('visibilitychange', handleVisibility);
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  const refreshWorkspace = useCallback(() => {
    readJson<WorkspaceInventory>('/api/workspace').then(setWorkspace).catch(() => undefined);
  }, []);

  function selectCategory(category: DeviceCategory) {
    if (!models.some((model) => model.category === category.id)) return;
    setCategoryId(category.id); setModelId(null); setError(null); setStep('model');
  }

  function selectModel(model: DeviceModelOption) {
    const nextMode = model.variants[0].mode;
    setModelId(model.id); setMode(nextMode); setCameras(defaultCameras(nextMode, model.category)); setError(null); setStep('hardware');
  }

  function updateMode(nextMode: ArmMode) {
    if (!selectedModel?.variants.some((variant) => variant.mode === nextMode)) return;
    setMode(nextMode); setCameras(defaultCameras(nextMode, categoryId ?? 'arm'));
  }

  function addCamera() {
    const camera = nextCameraDraft(cameras, mode, pendingCameraKind);
    if (camera) setCameras((current) => [...current, camera]);
  }

  function cameraSlotsFromDrafts() {
    return cameras.map((camera, index) => ({ alias: cameraSlotId(camera, cameras, index, mode), kind: camera.kind, side: (camera.kind === 'wrist' && mode === 'dual' ? camera.side ?? 'left' : 'single') as Side }));
  }

  async function saveDeclaration() {
    if (!selectedProfile) return;
    setBusy(true); setError(null);
    try {
      const configuration = await requestJson<DeviceConfiguration>('/api/config', 'PUT', { profile_id: selectedProfile.id, robot_type: selectedProfile.robot_type, teleoperator_type: selectedProfile.teleoperator_type, camera_slots: cameraSlotsFromDrafts(), serial_bindings: [], can_bindings: [], camera_bindings: [] });
      setSaved(configuration); setEditing(false); setActivePage('device'); setIdentifying(false);
    } catch (saveError) { setError(saveError instanceof Error ? saveError.message : '保存失败'); }
    finally { setBusy(false); }
  }

  const stopMotion = useCallback(async () => {
    if (!motionActive && !motionStarting) return;
    try { await postJson<{ status: string }>('/api/identify/motion/stop'); }
    finally { setMotionActive(false); setMotionStarting(false); }
  }, [motionActive, motionStarting]);

  const startMotion = useCallback(async (model: string, excludedIds: string[]) => {
    setMotionStarting(true);
    try {
      const result = await postJson<MotionStartResult>('/api/identify/motion/start', { model, excluded_ids: excludedIds });
      setMotionPorts(result.ports);
      if (result.readable_count === 0) throw new Error(result.ports.map((port) => port.motion_error).find(Boolean) || '未发现可识别的机械臂');
      setMotionActive(true);
    } finally { setMotionStarting(false); }
  }, []);

  const refreshCameras = useCallback(async () => {
    setCameraLoading(true);
    try { setCameraPreviews(await postJson<CameraPreview[]>('/api/identify/cameras')); }
    finally { setCameraLoading(false); }
  }, []);

  async function beginIdentification(scope: IdentificationScope = 'all') {
    if (!saved || !savedModel || !savedProfile) return;
    const preservedSerialAssignments = Object.fromEntries(saved.serial_bindings.map((binding) => [binding.alias, binding.id]));
    const preservedCanAssignments = Object.fromEntries(saved.can_bindings.map((binding) => [binding.alias, binding.id]));
    const preservedCameraAssignments = Object.fromEntries(cameras.flatMap((camera, index) => {
      const slot = saved.camera_slots[index];
      const binding = saved.camera_bindings.find((item) => item.alias === slot?.alias);
      return binding ? [[camera.id, binding.id]] : [];
    }));
    setBusy(true); setError(null); setIdentificationScope(scope);
    setSerialAssignments(scope === 'sensors' ? preservedSerialAssignments : {});
    setCanAssignments(scope === 'sensors' ? preservedCanAssignments : {});
    setCameraAssignments(scope === 'hardware' ? preservedCameraAssignments : {});
    setMotionPorts([]); setCameraPreviews([]); setIdentifying(true);
    try {
      const nextInventory = await readJson<HardwareInventory>('/api/devices');
      setInventory(nextInventory);
      if (scope !== 'hardware') setCameraPreviews(await postJson<CameraPreview[]>('/api/identify/cameras'));
      const firstSlot = hardwareSlots(savedProfile)[0];
      if (scope !== 'sensors' && firstSlot) await startMotion(savedModel.id, []);
    } catch (identifyError) { setError(identifyError instanceof Error ? identifyError.message : '设备识别启动失败'); }
    finally { setBusy(false); }
  }

  useEffect(() => {
    if (!identifying || activePage !== 'device' || !motionActive) return;
    const pendingSlots = hardwareSlots(savedProfile);
    let cancelled = false; let timer = 0;
    const poll = async () => {
      try {
        const result = await readJson<{ ports: MotionPort[] }>('/api/identify/motion/poll');
        if (cancelled) return;
        setMotionPorts(result.ports);
        const usedIds = [...Object.values(serialAssignments), ...Object.values(canAssignments)];
        const moved = result.ports.find((port) => port.moved && !usedIds.includes(port.stable_id));
        const pendingSlot = pendingSlots.find((slot) => !(slot.transport === 'socketcan' ? canAssignments[slot.id] : serialAssignments[slot.id]));
        if (moved && pendingSlot) {
          await postJson('/api/identify/motion/stop');
          if (cancelled) return;
          const nextSerialAssignments = pendingSlot.transport === 'serial' ? { ...serialAssignments, [pendingSlot.id]: moved.stable_id } : serialAssignments;
          const nextCanAssignments = pendingSlot.transport === 'socketcan' ? { ...canAssignments, [pendingSlot.id]: moved.stable_id } : canAssignments;
          setMotionActive(false); setSerialAssignments(nextSerialAssignments); setCanAssignments(nextCanAssignments);
          const nextSlot = pendingSlots.find((slot) => !(slot.transport === 'socketcan' ? nextCanAssignments[slot.id] : nextSerialAssignments[slot.id]));
          if (nextSlot && savedModel) await startMotion(savedModel.id, [...Object.values(nextSerialAssignments), ...Object.values(nextCanAssignments)]);
          return;
        }
        timer = window.setTimeout(poll, 650);
      } catch (pollError) {
        if (!cancelled) { setMotionActive(false); setError(pollError instanceof Error ? pollError.message : '机械臂识别中断'); }
      }
    };
    timer = window.setTimeout(poll, 350);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [activePage, canAssignments, identifying, motionActive, savedModel, savedProfile, serialAssignments, startMotion]);

  async function restartMotionIdentification() {
    await stopMotion(); setError(null); setSerialAssignments({}); setCanAssignments({}); setMotionPorts([]);
    if (savedModel && savedProfile && hardwareSlots(savedProfile).length > 0) {
      const firstSlot = hardwareSlots(savedProfile)[0];
      try { if (firstSlot) await startMotion(savedModel.id, []); }
      catch (motionError) { setError(motionError instanceof Error ? motionError.message : '机械臂识别启动失败'); }
    }
  }

  async function saveIdentification() {
    if (!saved || !savedProfile || !inventory) return;
    const pendingSlots = hardwareSlots(savedProfile);
    const serialPendingSlots = pendingSlots.filter((slot) => slot.transport === 'serial');
    const canPendingSlots = pendingSlots.filter((slot) => slot.transport === 'socketcan');
    if (serialPendingSlots.some((slot) => !serialAssignments[slot.id]) || canPendingSlots.some((slot) => !canAssignments[slot.id])) return setError('请完成机械臂识别');
    if (cameras.some((camera) => !cameraAssignments[camera.id])) return setError('请完成摄像头识别');
    setBusy(true);
    try {
    const serialBindings = identificationScope === 'sensors' ? saved.serial_bindings : serialPendingSlots.map((slot) => {
        const device = inventory.serial.find((item) => item.id === serialAssignments[slot.id]);
        if (!device) throw new Error(`未找到${slot.label}`);
        return { id: device.id, port: device.path, alias: slot.id, kind: slot.kind, side: slot.side };
      });
    const canBindings = identificationScope === 'sensors' ? saved.can_bindings : canPendingSlots.map((slot) => {
        const device = inventory.socketcan.find((item) => item.id === canAssignments[slot.id]);
        if (!device) throw new Error(`未找到${slot.label}`);
        return { id: device.id, alias: slot.id, kind: slot.kind, side: slot.side };
      });
    const cameraBindings = identificationScope === 'hardware' ? saved.camera_bindings : cameras.map((camera, index) => {
        const device = inventory.cameras.find((item) => item.id === cameraAssignments[camera.id]);
        if (!device) throw new Error(`未找到${cameraDisplayLabel(camera, cameras, index, mode)}`);
        const slot = saved.camera_slots[index];
        return {
          id: device.id, port: device.path, alias: slot.alias, side: slot.side,
          driver: device.driver, serial_number: device.serial_number || null,
        };
      });
      const configuration = await requestJson<DeviceConfiguration>('/api/config', 'PUT', { ...saved, serial_bindings: serialBindings, can_bindings: canBindings, camera_bindings: cameraBindings });
      setSaved(configuration); await stopMotion(); setIdentifying(false); setIdentificationScope('all');
    } catch (saveError) { setError(saveError instanceof Error ? saveError.message : '保存失败'); }
    finally { setBusy(false); }
  }

  function reconfigure() {
    if (!saved || !savedModel) return;
    const variant = savedModel.variants.find((item) => item.profile.id === saved.profile_id) ?? savedModel.variants[0];
    setCategoryId(savedModel.category); setModelId(savedModel.id); setMode(variant.mode); setCameras(cameraDraftsFromConfiguration(saved));
    setEditing(true); setStep('hardware'); setActivePage('device'); setError(null);
  }

  function navigate(page: PageId) {
    const available = page === 'device' || (page === 'maintenance' ? Boolean(saved) : ready);
    if (!available) return;
    if (page !== 'device') void stopMotion();
    setActivePage(page); setError(null);
  }

  const title = activePage === 'device' ? editing ? '新增设备' : '设备' : menu.find((item) => item.id === activePage)?.label ?? '';

  return <div className={`shell ${collapsed ? 'is-collapsed' : ''}`}>
    <aside>
      <div className="brand"><span className="brand-mark">E</span>{!collapsed && <strong>Evomind</strong>}<button className="collapse" type="button" onClick={() => setCollapsed((value) => !value)} aria-label={collapsed ? '展开菜单' : '收起菜单'}>{collapsed ? <PanelLeftOpen size={13} strokeWidth={1.8} /> : <PanelLeftClose size={13} strokeWidth={1.8} />}</button></div>
      <nav>{menu.map((item) => {
        const disabled = item.id === 'maintenance' ? !saved : item.id !== 'device' && !ready;
        return <button className={activePage === item.id ? 'active' : ''} type="button" disabled={disabled} onClick={() => navigate(item.id)} title={collapsed ? item.label : undefined} key={item.id}>{collapsed ? item.label.slice(0, 1) : item.label}</button>;
      })}</nav>
      <button className={`sound-control ${runtimeSounds.ready ? 'active' : ''}`} type="button" onClick={() => void runtimeSounds.toggle()} title={runtimeSounds.label} aria-label={runtimeSounds.label}>
        {runtimeSounds.ready ? <Volume2 size={16} /> : <VolumeX size={16} />}{!collapsed && <span>{runtimeSounds.label}</span>}
      </button>
      <div className="runtime"><i />{!collapsed && <div><strong>{status ? '运行正常' : '正在连接'}</strong><span>LeRobot {status?.lerobot_version ?? '—'}</span></div>}</div>
    </aside>
    <main>
      <header><div><p>EVOMIND / {status?.runtime.hostname ?? '本机'}</p><h1>{title}</h1></div>{(['teleoperation', 'recording', 'datasets', 'inference'] as PageId[]).includes(activePage) ? <div className="header-workflow-status" ref={setWorkflowStatusSlot} /> : activePage === 'device' && saved && !editing && !identifying && <div className="header-actions"><details className="header-recognition"><summary className="outline">重新识别 <ChevronDown className="recognition-chevron" size={15} strokeWidth={1.8} /></summary><div><button type="button" onClick={() => void beginIdentification('hardware')}>本体</button><button type="button" onClick={() => void beginIdentification('sensors')}>传感器</button><button type="button" onClick={() => void beginIdentification('all')}>全部设备</button></div></details><button className="outline" type="button" onClick={reconfigure}>重新配置</button></div>}</header>
      {error && <div className="error">{error}</div>}
      {activePage === 'device' && (!editing && saved ? ready && !identifying ? <DeviceOverview configuration={saved} model={savedModel} /> : !identifying ? <DeviceActivation model={savedModel} busy={busy} onStart={() => void beginIdentification()} /> : <IdentificationStep slots={hardwareSlots(savedProfile)} cameras={cameras} mode={mode} showHardware={identificationScope !== 'sensors'} showSensors={identificationScope !== 'hardware'} serialAssignments={serialAssignments} canAssignments={canAssignments} cameraAssignments={cameraAssignments} cameraPreviews={cameraPreviews} motionPorts={motionPorts} motionStarting={motionStarting} cameraLoading={cameraLoading || busy} busy={busy} onRestartMotion={() => void restartMotionIdentification()} onRefreshCameras={() => { setCameraAssignments({}); void refreshCameras().catch((cameraError) => setError(cameraError instanceof Error ? cameraError.message : '摄像头读取失败')); }} onCameraSelect={(cameraId, deviceId) => setCameraAssignments((current) => ({ ...current, [cameraId]: deviceId }))} onSave={() => void saveIdentification()} /> : <section className="device-setup">
        <div className="device-setup-steps">{createSteps.map((item, index) => {
          const disabled = item.id === 'model' ? !selectedCategory : item.id === 'hardware' ? !selectedModel : false;
          return <button className={`device-setup-step ${item.id === step ? 'active' : ''}`} type="button" disabled={disabled} onClick={() => setStep(item.id)} key={item.id}><span>{index + 1}</span>{item.label}</button>;
        })}</div>
        {step === 'category' && <div className="device-category-grid">{categories.map((category) => {
          const supported = models.some((model) => model.category === category.id);
          return <ChoiceCard title={category.title} icon={category.icon} active={category.id === categoryId} disabled={!supported} tooltip={supported ? undefined : '当前 LeRobot 未提供'} onClick={() => selectCategory(category)} key={category.id} />;
        })}</div>}
        {step === 'model' && <div className="device-model-grid">{categoryModels.map((model) => <ChoiceCard title={model.title} description={model.description} image={model.image} imageAlt={model.imageAlt} meta={model.category === 'arm' ? model.variants.map((variant) => armModeLabel(variant.mode)).join('、') : undefined} active={model.id === modelId} onClick={() => selectModel(model)} key={model.id} />)}</div>}
        {step === 'hardware' && selectedModel && selectedProfile && <HardwareStep model={selectedModel} mode={mode} cameras={cameras} pendingCameraKind={pendingCameraKind} onModeChange={updateMode} onCameraKindChange={setPendingCameraKind} onCameraSideChange={(cameraId, side) => setCameras((current) => current.map((camera) => camera.id === cameraId ? { ...camera, side } : camera))} onAddCamera={addCamera} onRemoveCamera={(id) => setCameras((current) => current.filter((camera) => camera.id !== id))} onSave={() => void saveDeclaration()} busy={busy} />}
      </section>)}
      {activePage === 'maintenance' && saved && <RepairPanel saved={saved} />}
      {activePage === 'calibration' && saved && <CalibrationPrototype configuration={saved} onConfigurationChange={setSaved} />}
      {activePage === 'teleoperation' && saved && <WorkflowPage kind="teleoperation" configuration={saved} workspace={workspace} runtimeEvent={runtimeEvent} storage={status?.runtime ?? null} statusSlot={workflowStatusSlot} onWorkspaceRefresh={refreshWorkspace} />}
      {activePage === 'recording' && saved && <WorkflowPage kind="recording" configuration={saved} workspace={workspace} runtimeEvent={runtimeEvent} storage={status?.runtime ?? null} statusSlot={workflowStatusSlot} onWorkspaceRefresh={refreshWorkspace} />}
      {activePage === 'collection-progress' && saved && <CollectionProgressPage runtimeEvent={runtimeEvent} policies={workspace.policies} />}
      {activePage === 'datasets' && saved && <DatasetViewerPage runtimeEvent={runtimeEvent} robotType={saved.robot_type} statusSlot={workflowStatusSlot} />}
      {activePage === 'inference' && saved && <WorkflowPage kind="inference" configuration={saved} workspace={workspace} runtimeEvent={runtimeEvent} storage={status?.runtime ?? null} statusSlot={workflowStatusSlot} onWorkspaceRefresh={refreshWorkspace} />}
    </main>
  </div>;
}

function HardwareStep({ model, mode, cameras, pendingCameraKind, onModeChange, onCameraKindChange, onCameraSideChange, onAddCamera, onRemoveCamera, onSave, busy }: { model: DeviceModelOption; mode: ArmMode; cameras: CameraDraft[]; pendingCameraKind: CameraKind; onModeChange: (mode: ArmMode) => void; onCameraKindChange: (kind: CameraKind) => void; onCameraSideChange: (cameraId: string, side: 'left' | 'right') => void; onAddCamera: () => void; onRemoveCamera: (id: string) => void; onSave: () => void; busy: boolean }) {
  return <div className="device-config-layout">
    {model.category === 'arm' && <section className="device-config-section"><div className="device-config-heading"><h3>本体形态</h3></div><div className="device-option-row">{model.variants.map((variant) => <button className={`device-option-card ${variant.mode === mode ? 'active' : ''}`} type="button" onClick={() => onModeChange(variant.mode)} key={variant.mode}><strong>{armModeLabel(variant.mode)}</strong></button>)}</div></section>}
    <section className="device-config-section"><div className="device-config-heading"><h3>摄像头</h3></div><div className="device-camera-content"><div className="device-camera-list">{cameras.map((camera, index) => <div className="device-camera-row declared" key={camera.id}><span>{index + 1}</span><strong>{cameraKindLabel(camera.kind)}</strong>{camera.kind === 'wrist' && mode === 'dual' && <div className="device-segmented device-side-segmented" role="group" aria-label="腕部标注">{(['left', 'right'] as const).map((side) => <button className={(camera.side ?? 'left') === side ? 'active' : ''} type="button" disabled={cameras.some((item) => item.id !== camera.id && item.kind === 'wrist' && item.side === side)} onClick={() => onCameraSideChange(camera.id, side)} key={side}>{side === 'left' ? '左腕' : '右腕'}</button>)}</div>}<button className="device-camera-remove" type="button" onClick={() => onRemoveCamera(camera.id)} aria-label="删除摄像头"><Trash2 size={16} /></button></div>)}</div><div className="device-camera-add-panel"><span>添加类型</span><div className="device-segmented">{(['wrist', 'environment'] as CameraKind[]).map((kind) => <button className={pendingCameraKind === kind ? 'active' : ''} type="button" onClick={() => onCameraKindChange(kind)} disabled={!canAddCameraKind(cameras, mode, kind)} key={kind}>{cameraKindLabel(kind)}</button>)}</div><button className="device-add-camera" type="button" onClick={onAddCamera} disabled={!canAddCameraKind(cameras, mode, pendingCameraKind)}><Plus size={15} />添加</button></div></div></section>
    <div className="device-setup-actions"><button className="primary" type="button" onClick={onSave} disabled={busy}>{busy ? '保存中' : '保存设备'}</button></div>
  </div>;
}

function IdentificationStep({ slots, cameras, mode, showHardware, showSensors, serialAssignments, canAssignments, cameraAssignments, cameraPreviews, motionPorts, motionStarting, cameraLoading, busy, onRestartMotion, onRefreshCameras, onCameraSelect, onSave }: { mode: ArmMode; slots: HardwareSlot[]; cameras: CameraDraft[]; showHardware: boolean; showSensors: boolean; serialAssignments: Record<string, string>; canAssignments: Record<string, string>; cameraAssignments: Record<string, string>; cameraPreviews: CameraPreview[]; motionPorts: MotionPort[]; motionStarting: boolean; cameraLoading: boolean; busy: boolean; onRestartMotion: () => void; onRefreshCameras: () => void; onCameraSelect: (cameraId: string, deviceId: string) => void; onSave: () => void }) {
  const assignment = (slot: HardwareSlot) => slot.transport === 'socketcan' ? canAssignments[slot.id] : serialAssignments[slot.id];
  const currentSlot = slots.find((slot) => !assignment(slot));
  const currentCamera = cameras.find((camera) => !cameraAssignments[camera.id]);
  const completed = !currentSlot && !currentCamera;
  const readablePorts = motionPorts.filter((port) => !port.motion_error).length;
  return <div className="identification-layout">
    {showHardware && slots.length > 0 && <section className="identify-section"><div className="device-config-heading with-action"><h3>本体</h3><button className="text-button" type="button" onClick={onRestartMotion}>重新识别</button></div><div className="identify-slots">{slots.map((slot) => {
      const deviceId = assignment(slot); const identified = Boolean(deviceId); const active = currentSlot?.id === slot.id;
      return <div className={`identify-slot ${active ? 'active' : ''}`} key={slot.id}><span className={`identify-dot ${identified ? 'complete' : ''}`}>{identified ? <Check size={14} /> : ''}</span><strong>{slot.label}</strong><small>{identified ? `已识别 · ${slot.transport === 'socketcan' ? deviceId : serialIdentity(deviceId)}` : active ? motionStarting ? '正在连接' : readablePorts ? '请轻轻移动这只机械臂' : '没有可读取的机械臂' : '等待识别'}</small></div>;
    })}</div></section>}
    {showSensors && cameras.length > 0 && <section className="identify-section"><div className="device-config-heading with-action"><h3>传感器</h3><button className="text-button inline-icon" type="button" onClick={onRefreshCameras} disabled={cameraLoading}><RefreshCw size={13} />{cameraLoading ? '读取中' : '重新识别'}</button></div>{currentCamera && <p className="identify-prompt">请选择 <strong>{cameraDisplayLabel(currentCamera, cameras, cameras.indexOf(currentCamera), mode)}</strong> 的画面</p>}<div className="identify-camera-grid">{cameraPreviews.map((camera) => {
      const assignedEntry = Object.entries(cameraAssignments).find(([, id]) => id === camera.id); const assignedCamera = cameras.find((item) => item.id === assignedEntry?.[0]); const assignedIndex = assignedCamera ? cameras.indexOf(assignedCamera) : -1;
      const available = Boolean(camera.preview_data_url);
      return <button className={`identify-camera-card ${assignedCamera ? 'used' : ''} ${available ? '' : 'unavailable'}`} type="button" disabled={!currentCamera || Boolean(assignedCamera) || !available} onClick={() => currentCamera && onCameraSelect(currentCamera.id, camera.id)} key={camera.id}>{camera.preview_data_url ? <img src={camera.preview_data_url} alt={camera.name} /> : <div className="identify-camera-placeholder"><strong>画面不可用</strong><small>{camera.preview_error}</small></div>}<span>{assignedCamera ? <><Check size={14} />{cameraDisplayLabel(assignedCamera, cameras, assignedIndex, mode)}</> : available ? '选择此画面' : camera.name}</span></button>;
    })}</div>{!cameraLoading && cameraPreviews.length === 0 && <p className="empty">未读取到摄像头画面</p>}</section>}
    <div className="device-setup-actions"><button className="primary" type="button" onClick={onSave} disabled={busy || !completed}>{busy ? '保存中' : '完成识别'}</button></div>
  </div>;
}

function RepairPanel({ saved }: { saved: DeviceConfiguration }) {
  const configuredTypes = [saved.robot_type, saved.teleoperator_type];
  const supportsFeetech = configuredTypes.some((type) => type && ['so100_follower', 'so100_leader', 'so101_follower', 'so101_leader', 'bi_so_follower', 'bi_so_leader'].includes(type));
  const supportsPiper = configuredTypes.some((type) => type && ['piperx_follower', 'piperx_leader', 'bi_piperx_follower', 'bi_piperx_leader'].includes(type));
  return <div className="repair-layout">{supportsFeetech ? <FeetechPanel saved={saved} /> : supportsPiper ? <PiperPanel saved={saved} /> : <p className="empty">当前设备暂未提供维修工具。</p>}</div>;
}

const piperControlModes: Record<number, string> = { 0: '待机', 1: 'CAN 控制', 2: '示教', 3: '以太网控制', 4: 'Wi-Fi 控制', 5: '遥控器控制', 6: '联动示教', 7: '离线轨迹' };
const piperArmStatuses: Record<number, string> = { 0: '正常', 1: '急停', 2: '无解', 3: '奇异点', 4: '目标超限', 5: '关节通信异常', 6: '抱闸未打开', 7: '碰撞保护', 8: '拖动超速', 9: '关节状态异常', 10: '其他异常', 14: '主控过温', 15: '释放电阻过温' };
const piperJointLimits: Record<number, [number, number]> = { 1: [-150, 150], 2: [0, 180], 3: [-170, 0], 4: [-100, 100], 5: [-70, 70], 6: [-120, 120] };

function piperCanLabel(device: CanDevice) {
  return `${device.interface} · ${device.serial_number || device.id}`;
}

function PiperPanel({ saved }: { saved: DeviceConfiguration }) {
  const [devices, setDevices] = useState<CanDevice[]>([]);
  const [deviceId, setDeviceId] = useState(saved.can_bindings[0]?.id ?? '');
  const [snapshot, setSnapshot] = useState<PiperSnapshot | null>(null);
  const [history, setHistory] = useState<PositionSample[]>([]);
  const [targets, setTargets] = useState<Record<number, number>>({});
  const [draggingId, setDraggingId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [actingId, setActingId] = useState<number | null>(null);
  const [controlArmed, setControlArmed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);
  const selectedDevice = devices.find((device) => device.id === deviceId);
  const scannedDeviceId = snapshot?.device_id;

  const loadDevices = useCallback(async () => {
    try {
      const inventory = await readJson<HardwareInventory>('/api/devices');
      setDevices(inventory.socketcan);
      setDeviceId((current) => inventory.socketcan.some((device) => device.id === current) ? current : inventory.socketcan[0]?.id ?? '');
    } catch { setError('读取 CAN 控制器失败'); }
  }, []);

  useEffect(() => {
    readJson<HardwareInventory>('/api/devices')
      .then((inventory) => {
        setDevices(inventory.socketcan);
        setDeviceId((current) => inventory.socketcan.some((device) => device.id === current) ? current : inventory.socketcan[0]?.id ?? '');
      })
      .catch(() => setError('读取 CAN 控制器失败'));
  }, []);

  useEffect(() => () => {
    void fetch('/api/maintenance/piper/close', { method: 'POST', keepalive: true });
  }, []);

  async function scanArm() {
    if (!deviceId) return;
    setBusy(true); setError(null); setLiveError(null); setControlArmed(false);
    try {
      const result = await postJson<PiperSnapshot>('/api/maintenance/piper/scan', { device_id: deviceId });
      setSnapshot(result);
      const positions = Object.fromEntries(result.motors.flatMap((motor) => motor.position === null ? [] : [[motor.id, motor.position]]));
      setTargets(positions); setHistory([{ capturedAt: Date.now(), positions }]);
    } catch (scanError) { setError(scanError instanceof Error ? scanError.message : 'PiperX 读取失败'); }
    finally { setBusy(false); }
  }

  useEffect(() => {
    if (!scannedDeviceId) return;
    let cancelled = false; let timer = 0;
    const poll = async () => {
      try {
        const result = await postJson<PiperSnapshot>('/api/maintenance/piper/snapshot', { device_id: scannedDeviceId });
        if (cancelled) return;
        setSnapshot(result); setLiveError(null);
        const positions = Object.fromEntries(result.motors.flatMap((motor) => motor.position === null ? [] : [[motor.id, motor.position]]));
        setTargets((current) => Object.fromEntries(result.motors.map((motor) => [motor.id, motor.id === draggingId ? current[motor.id] ?? motor.position ?? 0 : motor.position ?? current[motor.id] ?? 0])));
        setHistory((current) => [...current, { capturedAt: Date.now(), positions }].slice(-60));
        timer = window.setTimeout(poll, 300);
      } catch (pollError) {
        if (cancelled) return;
        setLiveError(pollError instanceof Error ? pollError.message : '实时读取中断');
        timer = window.setTimeout(poll, 1000);
      }
    };
    timer = window.setTimeout(poll, 300);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [draggingId, scannedDeviceId]);

  function armControl() {
    if (controlArmed) return setControlArmed(false);
    if (window.confirm('启用控制后可以使能 PiperX 关节。请扶稳机械臂并确认周围安全。')) setControlArmed(true);
  }

  async function action(motorId: number, actionName: 'enable' | 'disable' | 'move', value?: number) {
    if (!snapshot || (actionName !== 'disable' && !controlArmed)) return;
    setActingId(motorId); setError(null);
    try {
      const result = await postJson<PiperSnapshot>('/api/maintenance/piper/action', { device_id: snapshot.device_id, motor_id: motorId, action: actionName, value, confirmed: actionName !== 'disable' });
      setSnapshot(result);
    } catch (actionError) { setError(actionError instanceof Error ? actionError.message : 'PiperX 操作失败'); }
    finally { setActingId(null); }
  }

  const available = snapshot?.feedback_source !== 'none';
  const armStatus = snapshot?.status.available ? (piperArmStatuses[snapshot.status.arm_status] ?? `状态 ${snapshot.status.arm_status}`) : '未收到状态帧';
  return <section className="piper-panel">
    <div className="piper-toolbar">
      <div className="controller-field"><div className="field-label"><label htmlFor="piper-controller">CAN 控制器</label><button className="text-button inline-icon" type="button" onClick={() => void loadDevices()}><RefreshCw size={15} />刷新</button></div><select id="piper-controller" value={deviceId} onChange={(event) => { setDeviceId(event.target.value); setSnapshot(null); setHistory([]); setControlArmed(false); }}>{devices.map((device) => <option value={device.id} key={device.id}>{piperCanLabel(device)}</option>)}</select></div>
      <button className="primary inline-icon" type="button" onClick={() => void scanArm()} disabled={busy || !deviceId}><RefreshCw size={16} />{busy ? '读取中' : '读取机械臂'}</button>
    </div>
    {error && <div className="error compact">{error}</div>}
    {snapshot && <div className="piper-workspace">
      <div className="bus-heading"><div><p>{selectedDevice ? piperCanLabel(selectedDevice) : snapshot.device_id}</p><h3>PiperX</h3></div><div className="bus-actions"><span className={`live-state ${liveError ? 'failed' : ''}`}>{liveError ? '读取中断' : '实时读取中'}</span><button className={`outline ${controlArmed ? 'armed' : ''}`} type="button" onClick={armControl}>{controlArmed ? '锁定控制' : '启用控制'}</button></div></div>
      <div className="piper-summary">
        <div><span>固件</span><strong>{snapshot.firmware || '未读取'}</strong></div>
        <div><span>反馈</span><strong>{snapshot.feedback_source === 'feedback' ? '状态反馈' : snapshot.feedback_source === 'control' ? '主臂控制反馈' : '无反馈'}</strong></div>
        <div><span>控制模式</span><strong>{snapshot.status.available ? (piperControlModes[snapshot.status.ctrl_mode] ?? `模式 ${snapshot.status.ctrl_mode}`) : '—'}</strong></div>
        <div><span>机械臂状态</span><strong className={snapshot.status.arm_status ? 'failed-status' : ''}>{armStatus}</strong></div>
        <div><span>CAN 帧率</span><strong>{snapshot.can_fps.toFixed(0)} FPS</strong></div>
        <div><span>夹爪</span><strong>{snapshot.gripper.available ? `${snapshot.gripper.position.toFixed(1)} mm` : '未读取'}</strong></div>
      </div>
      {!available && <p className="empty">这个接口已连接，但暂未收到关节反馈。主臂静止时可能只在移动后产生控制反馈。</p>}
      <PiperPositionChart motors={snapshot.motors} history={history} />
      <div className="piper-bulk-actions"><button className="outline" type="button" onClick={() => void action(7, 'disable')} disabled={actingId !== null}>全部失能</button><button className="primary" type="button" onClick={() => void action(7, 'enable')} disabled={!controlArmed || actingId !== null}>全部使能</button></div>
      <div className="piper-motor-list">{snapshot.motors.map((motor) => {
        const [minimum, maximum] = piperJointLimits[motor.id]; const target = targets[motor.id] ?? motor.position ?? 0;
        return <div className="piper-motor-row" key={motor.id}>
        <div><strong>关节 {motor.id}</strong><span>{motor.position === null ? '—' : `${motor.position.toFixed(2)}°`}</span></div>
        <div className="position-control piper-position-control"><span>{minimum}</span><input type="range" min={minimum} max={maximum} step="0.1" value={target} disabled={!controlArmed || !motor.enabled || snapshot.feedback_source !== 'feedback' || actingId !== null} onPointerDown={() => setDraggingId(motor.id)} onChange={(event) => setTargets((current) => ({ ...current, [motor.id]: Number(event.target.value) }))} onPointerUp={(event) => { setDraggingId(null); void action(motor.id, 'move', Number(event.currentTarget.value)); }} onKeyUp={(event) => { if (event.key === 'Enter') void action(motor.id, 'move', Number(event.currentTarget.value)); }} /><span>{maximum}</span><output>{target.toFixed(1)}°</output></div>
        <div className="piper-telemetry"><span>{motor.voltage.toFixed(1)} V</span><span>{motor.current.toFixed(2)} A</span><span>驱动 {motor.driver_temperature} °C</span><span>电机 {motor.motor_temperature} °C</span></div>
        <div className={`piper-faults ${motor.faults.length ? 'failed' : ''}`}>{motor.faults.length ? motor.faults.join(' · ') : '状态正常'}</div>
        <button className={`torque-button ${motor.enabled ? 'enabled' : ''}`} type="button" disabled={actingId !== null || (!controlArmed && !motor.enabled)} onClick={() => void action(motor.id, motor.enabled ? 'disable' : 'enable')}>{actingId === motor.id ? '处理中' : motor.enabled ? '已使能' : '使能'}</button>
      </div>;})}</div>
    </div>}
  </section>;
}

function FeetechPanel({ saved }: { saved: DeviceConfiguration }) {
  const baudrates = [1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600, 38_400, 19_200];
  const [devices, setDevices] = useState<SerialDevice[]>([]);
  const [deviceId, setDeviceId] = useState(saved.serial_bindings[0]?.id ?? '');
  const [baudrate, setBaudrate] = useState(1_000_000);
  const [scan, setScan] = useState<FeetechScan | null>(null);
  const [busMotors, setBusMotors] = useState<{ id: number; model: string }[]>([]);
  const [history, setHistory] = useState<PositionSample[]>([]);
  const [targets, setTargets] = useState<Record<number, number>>({});
  const [draggingId, setDraggingId] = useState<number | null>(null);
  const [controlArmed, setControlArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [actingId, setActingId] = useState<number | null>(null);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [draftId, setDraftId] = useState('');
  const [devicesLoading, setDevicesLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);
  const selectedDevice = devices.find((device) => device.id === deviceId);
  const scannedDeviceId = scan?.device_id;
  const scannedBaudrate = scan?.baudrate;

  const loadDevices = useCallback(async () => {
    setDevicesLoading(true); setError(null); setNotice(null);
    try {
      const inventory = await readJson<HardwareInventory>('/api/devices');
      setDevices(inventory.serial);
      setDeviceId((current) => inventory.serial.some((device) => device.id === current) ? current : inventory.serial[0]?.id ?? '');
      setScan(null); setBusMotors([]); setHistory([]); setControlArmed(false); setEditingId(null);
    } catch { setError('读取控制器失败'); }
    finally { setDevicesLoading(false); }
  }, []);

  useEffect(() => {
    readJson<HardwareInventory>('/api/devices')
      .then((inventory) => {
        setDevices(inventory.serial);
        setDeviceId((current) => inventory.serial.some((device) => device.id === current) ? current : inventory.serial[0]?.id ?? '');
      })
      .catch(() => setError('读取控制器失败'));
  }, []);

  function applyScan(result: FeetechScan) {
    setScan(result);
    setBusMotors(result.motors.map((motor) => ({ id: motor.id, model: motor.model })));
    const positions = Object.fromEntries(result.motors.map((motor) => [motor.id, motor.position]));
    setTargets(positions);
    setHistory(result.motors.length ? [{ capturedAt: Date.now(), positions }] : []);
    setControlArmed(false); setLiveError(null);
  }

  async function refresh() {
    if (!deviceId) return;
    setBusy(true); setError(null); setNotice(null); setEditingId(null);
    try {
      const result = await postJson<FeetechScan>('/api/maintenance/feetech/scan', { device_id: deviceId, baudrate });
      applyScan(result);
    } catch (scanError) { setError(scanError instanceof Error ? scanError.message : '扫描失败'); }
    finally { setBusy(false); }
  }

  useEffect(() => {
    if (!scannedDeviceId || !scannedBaudrate || !busMotors.length) return;
    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      try {
        const snapshot = await postJson<FeetechSnapshot>('/api/maintenance/feetech/snapshot', {
          device_id: scannedDeviceId,
          baudrate: scannedBaudrate,
          motors: busMotors,
        });
        if (cancelled) return;
        const positions = Object.fromEntries(snapshot.positions.map((item) => [item.id, item.position]));
        setScan((current) => current ? { ...current, motors: current.motors.map((motor) => ({ ...motor, position: positions[motor.id] ?? motor.position })) } : current);
        setTargets((current) => Object.fromEntries(Object.entries(current).map(([id, target]) => [id, Number(id) === draggingId ? target : positions[Number(id)] ?? target])));
        setHistory((current) => [...current, { capturedAt: Date.now(), positions }].slice(-60));
        setLiveError(null);
        timer = window.setTimeout(poll, 200);
      } catch (pollError) {
        if (cancelled) return;
        setLiveError(pollError instanceof Error ? pollError.message : '实时读取中断');
        timer = window.setTimeout(poll, 800);
      }
    };
    timer = window.setTimeout(poll, 200);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [busMotors, draggingId, scannedBaudrate, scannedDeviceId]);

  function armControl() {
    if (controlArmed) return setControlArmed(false);
    if (window.confirm('启用后可开启扭矩并移动机械臂。请确认机械臂周围安全。')) setControlArmed(true);
  }

  async function action(motorId: number, name: 'torque_enable' | 'torque_disable' | 'move', value?: number) {
    if (!deviceId) return;
    if (name !== 'torque_disable' && !controlArmed) return setError('请先启用控制');
    setActingId(motorId); setError(null);
    try {
      const result = await postJson<FeetechScan>('/api/maintenance/feetech/action', { device_id: deviceId, baudrate, motor_id: motorId, action: name, value, confirmed: name === 'move' });
      setScan(result);
    } catch (actionError) { setError(actionError instanceof Error ? actionError.message : '操作失败'); }
    finally { setActingId(null); }
  }

  function editMotorId(motorId: number) {
    setEditingId(motorId); setDraftId(String(motorId)); setError(null); setNotice(null);
  }

  async function changeMotorId(motor: FeetechMotor) {
    if (!deviceId || !scan) return;
    const nextId = Number(draftId);
    if (!Number.isInteger(nextId) || nextId < 0 || nextId > 253) return setError('舵机 ID 必须是 0–253 之间的整数');
    if (nextId === motor.id) return setError('新 ID 必须与当前 ID 不同');
    if (scan.motors.some((item) => item.id === nextId)) return setError(`ID ${nextId} 已被总线上的其他舵机使用`);
    if (!window.confirm(`确定把舵机 ID ${motor.id} 修改为 ID ${nextId}？\n\n写入 EEPROM 时会关闭该舵机扭矩。请确保供电稳定、目标 ID 未被占用。SO-101 标准关节通常使用 ID 1–6。`)) return;

    setActingId(motor.id); setError(null); setNotice(null); setBusMotors([]);
    try {
      const result = await postJson<FeetechIdChange>('/api/maintenance/feetech/action', {
        device_id: deviceId, baudrate, motor_id: motor.id, action: 'set_id', value: nextId, confirmed: true,
      });
      const rescanned = await postJson<FeetechScan>('/api/maintenance/feetech/scan', { device_id: deviceId, baudrate });
      applyScan(rescanned); setEditingId(null);
      setNotice(`ID ${result.old_id} 已修改为 ID ${result.new_id}，并已重新扫描总线。`);
    } catch (actionError) {
      const message = actionError instanceof Error ? actionError.message : '修改舵机 ID 失败';
      try {
        const rescanned = await postJson<FeetechScan>('/api/maintenance/feetech/scan', { device_id: deviceId, baudrate });
        applyScan(rescanned);
      } catch { setScan(null); }
      setError(message);
      setEditingId(null);
    } finally { setActingId(null); }
  }

  return <section className="feetech-panel">
    <div className="feetech-toolbar">
      <div className="controller-field"><div className="field-label"><label htmlFor="feetech-controller">控制器</label><button className="text-button inline-icon" type="button" onClick={() => void loadDevices()} disabled={devicesLoading}><RefreshCw size={15} />{devicesLoading ? '刷新中' : '刷新'}</button></div><select id="feetech-controller" value={deviceId} onChange={(event) => { setDeviceId(event.target.value); setScan(null); setBusMotors([]); setHistory([]); setEditingId(null); setNotice(null); }}>{devices.map((device) => <option value={device.id} title={device.id} key={device.id}>{serialLabel(device)}</option>)}</select></div>
      <label>波特率<select value={baudrate} onChange={(event) => { setBaudrate(Number(event.target.value)); setScan(null); setBusMotors([]); setHistory([]); setEditingId(null); setNotice(null); }}>{baudrates.map((value) => <option value={value} key={value}>{value.toLocaleString()}</option>)}</select></label>
      <button className="primary inline-icon" type="button" onClick={() => void refresh()} disabled={busy || !deviceId}><RefreshCw size={16} />{busy ? '扫描中' : '扫描总线'}</button>
    </div>
    {notice && <div className="success compact">{notice}</div>}
    {error && <div className="error compact">{error}</div>}
    {scan && scan.motors.length === 0 && <p className="empty">这个控制器下没有发现飞特舵机。</p>}
    {scan && scan.motors.length > 0 && <div className="bus-workspace">
      <div className="bus-heading"><div><p>{selectedDevice ? serialLabel(selectedDevice) : scan.device_id}</p><h3>{scan.motors.length} 个舵机</h3></div><div className="bus-actions"><span className={`live-state ${liveError ? 'failed' : ''}`}>{liveError ? '读取中断' : '实时读取中'}</span><button className={`outline ${controlArmed ? 'armed' : ''}`} type="button" onClick={armControl}>{controlArmed ? '锁定控制' : '启用控制'}</button></div></div>
      <PositionChart motors={scan.motors} history={history} />
      <div className="servo-control-list">{scan.motors.map((motor) => {
        const target = targets[motor.id] ?? motor.position;
        const working = actingId === motor.id;
        return <div className="servo-control-row" key={motor.id}>
          <div className="servo-row-title"><div><strong>ID {motor.id}</strong><span>{motor.model}</span></div><div className="servo-readouts"><span>{motor.position}</span><small>{motor.temperature} °C · {motor.voltage.toFixed(1)} V</small></div></div>
          <div className="position-control"><span>0</span><input type="range" min="0" max="4095" step="1" value={target} disabled={!controlArmed || !motor.torque_enabled || working} onPointerDown={() => setDraggingId(motor.id)} onChange={(event) => setTargets((current) => ({ ...current, [motor.id]: Number(event.target.value) }))} onPointerUp={(event) => { setDraggingId(null); void action(motor.id, 'move', Number(event.currentTarget.value)); }} onKeyUp={(event) => { if (event.key === 'Enter') void action(motor.id, 'move', Number(event.currentTarget.value)); }} /><span>4095</span><output>{target}</output></div>
          <div className="servo-row-actions">
            <button className="outline servo-id-button" type="button" disabled={actingId !== null} onClick={() => editMotorId(motor.id)}>修改 ID</button>
            <button className={`torque-button ${motor.torque_enabled ? 'enabled' : ''}`} type="button" disabled={working || (!controlArmed && !motor.torque_enabled)} onClick={() => void action(motor.id, motor.torque_enabled ? 'torque_disable' : 'torque_enable')}>{working ? '处理中' : motor.torque_enabled ? '扭矩已开启' : '开启扭矩'}</button>
          </div>
          {editingId === motor.id && <div className="servo-id-editor">
            <label htmlFor={`servo-id-${motor.id}`}>新 ID</label>
            <input id={`servo-id-${motor.id}`} type="number" min="0" max="253" step="1" value={draftId} onChange={(event) => setDraftId(event.target.value)} autoFocus />
            <span>目标 ID 必须空闲；写入时会关闭该舵机扭矩。SO-101 标准关节通常使用 ID 1–6。</span>
            <button className="outline" type="button" onClick={() => setEditingId(null)}>取消</button>
            <button className="primary" type="button" disabled={working} onClick={() => void changeMotorId(motor)}>{working ? '写入中' : '写入并重新扫描'}</button>
          </div>}
        </div>;
      })}</div>
    </div>}
  </section>;
}

const chartColors = ['#2e7652', '#db7a4d', '#4f78c4', '#9a63b5', '#d2a52d', '#3e9ca7', '#c65368', '#6f7c72'];

function PiperPositionChart({ motors, history }: { motors: PiperMotor[]; history: PositionSample[] }) {
  const width = 920; const height = 260; const left = 48; const top = 18; const plotWidth = width - left - 16; const plotHeight = height - top - 32;
  const minimum = -180; const maximum = 180; const yTicks = [-180, -90, 0, 90, 180];
  const yFor = (value: number) => top + plotHeight - ((value - minimum) / (maximum - minimum)) * plotHeight;
  return <section className="position-chart"><div className="chart-title"><div><h3>实时关节角度</h3><span>度</span></div><div className="chart-legend">{motors.map((motor, index) => <span key={motor.id}><i style={{ background: chartColors[index % chartColors.length] }} />关节 {motor.id}</span>)}</div></div><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="PiperX 关节实时角度折线图" preserveAspectRatio="none">
    {yTicks.map((tick) => <g key={tick}><line x1={left} x2={width - 16} y1={yFor(tick)} y2={yFor(tick)} className="chart-grid-line" /><text x={left - 9} y={yFor(tick) + 4} textAnchor="end">{tick}</text></g>)}
    {motors.map((motor, motorIndex) => {
      const points = history.flatMap((sample, index) => {
        const value = sample.positions[motor.id];
        if (value === undefined) return [];
        const x = left + (history.length <= 1 ? plotWidth : (index / (history.length - 1)) * plotWidth);
        return [`${x},${yFor(value)}`];
      }).join(' ');
      return <polyline key={motor.id} points={points} fill="none" stroke={chartColors[motorIndex % chartColors.length]} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />;
    })}
  </svg></section>;
}

function PositionChart({ motors, history }: { motors: FeetechMotor[]; history: PositionSample[] }) {
  const width = 920; const height = 260; const left = 48; const top = 18; const plotWidth = width - left - 16; const plotHeight = height - top - 32;
  const yTicks = [0, 1024, 2048, 3072, 4095];
  return <section className="position-chart"><div className="chart-title"><div><h3>实时位置</h3><span>0–4095</span></div><div className="chart-legend">{motors.map((motor, index) => <span key={motor.id}><i style={{ background: chartColors[index % chartColors.length] }} />ID {motor.id}</span>)}</div></div><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="舵机实时位置折线图" preserveAspectRatio="none">
    {yTicks.map((tick) => {
      const y = top + plotHeight - (tick / 4095) * plotHeight;
      return <g key={tick}><line x1={left} x2={width - 16} y1={y} y2={y} className="chart-grid-line" /><text x={left - 9} y={y + 4} textAnchor="end">{tick}</text></g>;
    })}
    {motors.map((motor, motorIndex) => {
      const points = history.map((sample, index) => {
        const x = left + (history.length <= 1 ? plotWidth : (index / (history.length - 1)) * plotWidth);
        const y = top + plotHeight - ((sample.positions[motor.id] ?? motor.position) / 4095) * plotHeight;
        return `${x},${y}`;
      }).join(' ');
      return <polyline key={motor.id} points={points} fill="none" stroke={chartColors[motorIndex % chartColors.length]} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />;
    })}
  </svg></section>;
}
function ChoiceCard({ active, title, description, image, imageAlt, meta, tooltip, icon: Icon, disabled = false, onClick }: { active: boolean; title: string; description?: string; image?: string; imageAlt?: string; meta?: string; tooltip?: string; icon?: LucideIcon; disabled?: boolean; onClick: () => void }) { return <button className={`device-template-card ${active ? 'active' : ''} ${disabled ? 'unsupported' : ''}`} type="button" onClick={onClick} disabled={disabled} title={tooltip}>{Icon && <Icon size={22} />}{image && <div className="device-model-visual"><img src={image} alt={imageAlt ?? ''} /></div>}<strong>{title}</strong>{description && <span>{description}</span>}{meta && <small>{meta}</small>}</button>; }

function DeviceActivation({ model, busy, onStart }: { model?: DeviceModelOption; busy: boolean; onStart: () => void }) {
  return <section className="activation-view"><div><span className="eyebrow">设备已保存</span><h2>{model?.title ?? '当前设备'}</h2><p>完成机械臂和摄像头识别后即可使用全部功能。</p></div><button className="primary" type="button" onClick={onStart} disabled={busy}>{busy ? '正在启动' : '开始识别'}</button></section>;
}

function bindingTitle(alias: string) {
  const labels: Record<string, string> = {
    left_leader_arm: '左主臂', left_follower_arm: '左从臂', right_leader_arm: '右主臂', right_follower_arm: '右从臂',
    leader_arm: '主臂', follower_arm: '从臂', left_wrist: '左腕摄像头', right_wrist: '右腕摄像头', environment_1: '环境摄像头',
  };
  return labels[alias] ?? alias.replaceAll('_', ' ');
}

function DeviceOverview({ configuration, model }: { configuration: DeviceConfiguration; model?: DeviceModelOption }) {
  const [inventory, setInventory] = useState<HardwareInventory | null>(null);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => readJson<HardwareInventory>('/api/devices')
      .then((nextInventory) => { if (!cancelled) setInventory(nextInventory); })
      .catch(() => { if (!cancelled) setInventory(null); });
    void refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);

  const serialIds = new Set(inventory?.serial.map((device) => device.id) ?? []);
  const canDevices = new Map(inventory?.socketcan.map((device) => [device.id, device]) ?? []);
  const cameraIds = new Set(inventory?.cameras.map((device) => device.id) ?? []);
  const bindings = [
    ...configuration.serial_bindings.map((binding) => ({ id: binding.id, alias: binding.alias, kind: binding.kind === 'robot' ? '机械臂' : '遥操作设备', port: binding.port, online: serialIds.has(binding.id) })),
    ...configuration.can_bindings.map((binding) => ({ id: binding.id, alias: binding.alias, kind: binding.kind === 'robot' ? '机械臂 · CAN' : '遥操作设备 · CAN', port: canDevices.get(binding.id)?.interface ?? binding.id, online: canDevices.has(binding.id) })),
    ...configuration.camera_bindings.map((binding) => ({ id: binding.id, alias: binding.alias, kind: '摄像头', port: binding.port, online: cameraIds.has(binding.id) })),
  ];
  const offlineCount = inventory ? bindings.filter((binding) => !binding.online).length : 0;
  return <section className="device-overview">
    <div className="overview-hero"><div className={offlineCount ? 'offline-status' : ''}><span className="status-dot" />{inventory ? offlineCount ? `${offlineCount} 个接口未连接` : '全部接口在线' : '正在检查接口'}</div><h2>{model?.title ?? configuration.profile_id}</h2><p>{configuration.robot_type}{configuration.teleoperator_type ? ` + ${configuration.teleoperator_type}` : ''}</p></div>
    <div className="data-list"><div className="data-list-head"><span>设备</span><span>类型</span><span>接口</span><span>状态</span></div>{bindings.map((binding) => <div className="data-row" key={`${binding.alias}-${binding.id}`}><strong>{bindingTitle(binding.alias)}</strong><span>{binding.kind}</span><code title={binding.port}>{binding.port}</code><span className={`row-status${inventory && !binding.online ? ' offline' : ''}`}><i />{inventory ? binding.online ? '在线' : '未连接' : '检查中'}</span></div>)}</div>
  </section>;
}

function CalibrationPrototype({ configuration, onConfigurationChange }: { configuration: DeviceConfiguration; onConfigurationChange: (configuration: DeviceConfiguration) => void }) {
  if (configuration.can_bindings.length > 0 && configuration.serial_bindings.length === 0) {
    return <section className="workspace-view"><div className="activation-view"><div><span className="eyebrow">绝对编码器</span><h2>不需要校准</h2><p>PiperX 使用绝对编码器，设备连接后可直接使用。</p></div></div></section>;
  }
  return <SerialCalibrationPrototype configuration={configuration} onConfigurationChange={onConfigurationChange} />;
}

function SerialCalibrationPrototype({ configuration, onConfigurationChange }: { configuration: DeviceConfiguration; onConfigurationChange: (configuration: DeviceConfiguration) => void }) {
  const [status, setStatus] = useState<CalibrationStatus | null>(null);
  const [error, setError] = useState('');
  const refreshedRun = useRef(0);
  const active = status?.state === 'starting' || status?.state === 'running' || status?.state === 'stopping';

  const refresh = useCallback(async () => {
    const next = await readJson<CalibrationStatus>('/api/calibration/status');
    setStatus(next);
    if (next.state === 'done' && next.updated_at !== refreshedRun.current) {
      refreshedRun.current = next.updated_at;
      onConfigurationChange(await readJson<DeviceConfiguration>('/api/config'));
    }
  }, [onConfigurationChange]);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try { await refresh(); }
      catch (pollError) { if (!cancelled) setError(pollError instanceof Error ? pollError.message : '无法读取校准状态'); }
    };
    void poll();
    return () => { cancelled = true; };
  }, [refresh]);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      void refresh().catch((pollError) => setError(pollError instanceof Error ? pollError.message : '无法读取校准状态'));
    }, 700);
    return () => window.clearInterval(timer);
  }, [active, refresh]);

  async function startManual(alias: string) {
    setError('');
    if (!window.confirm('手动校准需要移动机械臂所有关节。请确认周围空间安全。')) return;
    try { setStatus(await postJson<CalibrationStatus>('/api/calibration/manual/start', { alias })); }
    catch (startError) { setError(startError instanceof Error ? startError.message : '校准启动失败'); }
  }

  async function startAuto(alias?: string) {
    setError('');
    const message = alias
      ? `${bindingTitle(alias)}将自动移动。请清空周围空间并确认供电稳定。`
      : '所有尚未启动的机械臂将同时自动移动。请清空机械臂周围空间并确认供电稳定。';
    if (!window.confirm(message)) return;
    try { setStatus(await postJson<CalibrationStatus>('/api/calibration/auto/start', alias ? { alias } : {})); }
    catch (startError) { setError(startError instanceof Error ? startError.message : '自动校准启动失败'); }
  }

  async function advanceManual() {
    setError('');
    try { setStatus(await postJson<CalibrationStatus>('/api/calibration/manual/advance')); }
    catch (advanceError) { setError(advanceError instanceof Error ? advanceError.message : '无法进入下一步'); }
  }

  async function stop() {
    setError('');
    try { setStatus(await postJson<CalibrationStatus>('/api/calibration/stop')); }
    catch (stopError) { setError(stopError instanceof Error ? stopError.message : '自动校准停止失败'); }
  }

  function supportsAuto(binding: SerialBinding) {
    const type = binding.kind === 'robot' ? configuration.robot_type : configuration.teleoperator_type ?? '';
    return ['so101_follower', 'so101_leader', 'bi_so_follower', 'bi_so_leader'].includes(type);
  }

  function supportsManual(binding: SerialBinding) {
    const type = binding.kind === 'robot' ? configuration.robot_type : configuration.teleoperator_type ?? '';
    return ['so100_follower', 'so100_leader', 'so101_follower', 'so101_leader', 'bi_so_follower', 'bi_so_leader'].includes(type);
  }

  const manualAction = status?.mode === 'manual' && status.prompt_id === 'calibration_middle'
    ? '已放到中位'
    : status?.mode === 'manual' && status.phase === 'recording_ranges' ? '完成范围记录' : '';

  const automaticCount = configuration.serial_bindings.filter(supportsAuto).length;
  const remainingAutomaticCount = active
    ? configuration.serial_bindings.filter((binding) => supportsAuto(binding) && !status?.devices[binding.alias]?.run).length
    : automaticCount;
  const autoBatchJoinable = !active || status?.mode === 'auto';

  return <section className="workspace-view">
    {error && <div className="error compact">{error}</div>}
    {automaticCount > 1 && <div className="calibration-batch-actions"><button className="primary" type="button" disabled={!autoBatchJoinable || remainingAutomaticCount === 0 || status?.state === 'stopping'} onClick={() => void startAuto()}>并行自动校准</button></div>}
    {status && status.state !== 'idle' && <div className={`calibration-progress ${status.state}`}><div><span>{status.mode === 'auto' ? '自动校准' : status.alias ? bindingTitle(status.alias) : '校准'}</span><strong>{status.message || '等待开始'}</strong>{status.motor && <small>当前关节：{status.motor}</small>}</div>{active && <div className="calibration-progress-actions">{manualAction && <button className="primary" type="button" onClick={() => void advanceManual()}>{manualAction}</button>}<button className="outline" type="button" onClick={() => void stop()} disabled={status.state === 'stopping'}>{status.state === 'stopping' ? '正在停止' : '停止'}</button></div>}</div>}
    <div className="action-list">{configuration.serial_bindings.map((binding) => {
      const run = status?.devices[binding.alias]?.run;
      const running = run?.state === 'starting' || run?.state === 'running' || run?.state === 'stopping' || (active && status?.mode === 'manual' && status.alias === binding.alias);
      const automatic = supportsAuto(binding);
      const manual = supportsManual(binding);
      const calibrated = status?.devices[binding.alias]?.available ?? false;
      const runFailed = run?.state === 'error';
      const autoDisabled = !automatic || !autoBatchJoinable || (active && Boolean(run)) || status?.state === 'stopping';
      return <div className="action-row" key={binding.id}><div><strong>{bindingTitle(binding.alias)}</strong><span>{binding.kind === 'robot' ? '机械臂' : '遥操作设备'}</span></div><code>{serialIdentity(binding.id)}</code><span className={runFailed ? 'failed-status' : calibrated ? 'calibrated-status' : 'muted-status'} title={run?.error || ''}>{running ? run?.motor ? `校准中 · ${run.motor}` : '校准中' : runFailed ? '校准失败' : calibrated ? '有校准文件' : '未校准'}</span><div className="calibration-actions"><button className="outline" type="button" disabled={active || !manual} onClick={() => void startManual(binding.alias)}>手动校准</button>{automatic && <button className="outline" type="button" disabled={autoDisabled} onClick={() => void startAuto(binding.alias)}>自动校准</button>}</div></div>;
    })}</div>
  </section>;
}

function WorkflowPage({ kind, configuration, workspace, runtimeEvent, storage, statusSlot, onWorkspaceRefresh }: { kind: 'teleoperation' | 'recording' | 'inference' | 'replay'; configuration: DeviceConfiguration; workspace: WorkspaceInventory; runtimeEvent: RuntimeEvent | null; storage: StorageInfo | null; statusSlot: HTMLDivElement | null; onWorkspaceRefresh: () => void }) {
  const followers = [...configuration.serial_bindings, ...configuration.can_bindings].filter((binding) => binding.kind === 'robot');
  const [fps, setFps] = useState(30);
  const [task, setTask] = useState('Insert the copper screw into the black sleeve');
  const [taskId, setTaskId] = useState('');
  const [dailyTasks, setDailyTasks] = useState<DailyCollectionTask[]>([]);
  const [duration, setDuration] = useState(120);
  const [inference, setInference] = useState<RolloutInference>('sync');
  const [policyPath, setPolicyPath] = useState(workspace.policies[0]?.path ?? '');
  const [policyInspection, setPolicyInspection] = useState<PolicyInspection | null>(null);
  const [policyInspecting, setPolicyInspecting] = useState(false);
  const [policyPreloading, setPolicyPreloading] = useState(false);
  const [datasetId, setDatasetId] = useState(workspace.datasets[0]?.id ?? '');
  const [episode, setEpisode] = useState(0);
  const [runtime, setRuntime] = useState<WorkflowRuntime>({ running: false, job_id: null, operation: null, event: null });
  const [operationError, setOperationError] = useState('');
  const [pendingCommand, setPendingCommand] = useState<'stop' | 'finish_episode' | 'rerecord_episode' | 'pause_resume' | 'correction' | 'toggle_highlight' | null>(null);
  const pendingSequence = useRef(0);
  const wasRunning = useRef(false);

  const refreshTasks = useCallback(() => {
    if (kind !== 'recording') return Promise.resolve();
    return Promise.all([
      readJson<DailyCollectionTask[]>('/api/collection/tasks'),
      readJson<{ active_session: { task_id: string } | null }>('/api/collection/progress'),
    ])
      .then(([tasks, progress]) => {
        setDailyTasks(tasks);
        setTaskId((current) => progress.active_session?.task_id
          ?? (tasks.some((item) => item.id === current) ? current : tasks[0]?.id ?? ''));
      })
      .catch(() => setDailyTasks([]));
  }, [kind]);

  useEffect(() => {
    const refresh = () => readJson<WorkflowRuntime>('/api/runtime/status').then(setRuntime).catch(() => undefined);
    void refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => { void refreshTasks(); }, [refreshTasks]);

  useEffect(() => {
    let timer: number | undefined;
    try {
      const remembered = Number(window.localStorage.getItem(`evomind-lerobot:${kind}:fps`));
      if ([15, 20, 30].includes(remembered)) timer = window.setTimeout(() => setFps(remembered), 0);
    } catch { /* Use the default FPS when local storage is unavailable. */ }
    return () => { if (timer !== undefined) window.clearTimeout(timer); };
  }, [kind]);

  useEffect(() => {
    if (wasRunning.current && !runtime.running) { onWorkspaceRefresh(); void refreshTasks(); }
    wasRunning.current = runtime.running;
  }, [onWorkspaceRefresh, refreshTasks, runtime.running]);

  const selectedTask = dailyTasks.find((item) => item.id === taskId);
  const selectedStrategy = selectedTask?.collection_method === 'policy' ? selectedTask.rollout_strategy : null;
  const operation = kind === 'recording'
    ? runtime.running && (runtime.operation === 'recording' || runtime.operation === 'rollout')
      ? runtime.operation
      : selectedTask?.collection_method === 'policy' ? 'rollout' : 'recording'
    : kind === 'inference' ? 'rollout' : kind;
  const endpoint = kind === 'recording'
    ? '/api/runtime/collection/start'
    : kind === 'inference' ? '/api/runtime/rollout/start' : `/api/runtime/${kind}/start`;
  const effectivePolicyPath = kind === 'recording' ? selectedTask?.policy_path ?? '' : policyPath || workspace.policies[0]?.path || '';
  const effectiveDatasetId = datasetId || workspace.datasets[0]?.id || '';
  const selectedDataset = workspace.datasets.find((item) => item.id === effectiveDatasetId);
  const selectedPolicy = workspace.policies.find((item) => item.path === effectivePolicyPath);
  const runningThis = runtime.running && runtime.operation === operation;
  const runningOther = runtime.running && !runningThis;
  const event = newestRuntimeEvent(
    runtimeEvent?.operation === operation ? runtimeEvent : null,
    runtime.event?.operation === operation ? runtime.event : null,
  );
  const visibleError = operationError || (event?.phase === 'failed' ? event.message : '');
  const rolloutPhase = typeof event?.data.rollout_phase === 'string' ? event.data.rollout_phase : 'autonomous';
  const recordingPhase = event?.phase;

  useEffect(() => {
    if (pendingCommand && event && event.sequence > pendingSequence.current) setPendingCommand(null);
  }, [event, pendingCommand]);

  function updateFps(value: number) {
    setFps(value);
    try { window.localStorage.setItem(`evomind-lerobot:${kind}:fps`, String(value)); } catch { /* Keep the selector usable without persistence. */ }
  }

  async function start() {
    setOperationError('');
    if ((kind === 'inference' || selectedTask?.collection_method === 'policy') && !window.confirm('Policy 会直接驱动机械臂。请确认急停可用、周围空间已清空，并让操作员随时准备接管。')) return;
    let body: Record<string, unknown> = { fps };
    if (kind === 'recording') body = { task_id: taskId };
    if (kind === 'inference') body = { policy_path: effectivePolicyPath, strategy: 'base', inference, task, fps, duration_s: duration };
    if (kind === 'replay') body = { dataset_id: effectiveDatasetId, episode };
    try { setRuntime(await postJson<WorkflowRuntime>(endpoint, body)); }
    catch (startError) { setOperationError(startError instanceof Error ? startError.message : '启动失败'); }
  }

  async function inspectPolicy() {
    setOperationError(''); setPolicyInspecting(true);
    try { setPolicyInspection(await postJson<PolicyInspection>('/api/runtime/policy/inspect', { policy_path: effectivePolicyPath })); }
    catch (inspectError) { setPolicyInspection(null); setOperationError(inspectError instanceof Error ? inspectError.message : '模型检查失败'); }
    finally { setPolicyInspecting(false); }
  }

  async function preloadPolicy() {
    setOperationError(''); setPolicyPreloading(true);
    try {
      const residency = await postJson<PolicyResidency>('/api/runtime/policy/preload', { policy_path: effectivePolicyPath });
      setRuntime((current) => ({ ...current, policy_residency: residency }));
    } catch (preloadError) {
      setOperationError(preloadError instanceof Error ? preloadError.message : '模型预加载失败');
    } finally { setPolicyPreloading(false); }
  }

  async function unloadPolicy() {
    setOperationError('');
    try {
      const residency = await postJson<PolicyResidency>('/api/runtime/policy/unload');
      setRuntime((current) => ({ ...current, policy_residency: residency }));
    } catch (unloadError) { setOperationError(unloadError instanceof Error ? unloadError.message : '模型卸载失败'); }
  }

  async function reloadPolicy() {
    setOperationError(''); setPolicyPreloading(true);
    try {
      const empty = await postJson<PolicyResidency>('/api/runtime/policy/unload');
      setRuntime((current) => ({ ...current, policy_residency: empty }));
      const residency = await postJson<PolicyResidency>('/api/runtime/policy/preload', { policy_path: effectivePolicyPath });
      setRuntime((current) => ({ ...current, policy_residency: residency }));
    } catch (reloadError) {
      setOperationError(reloadError instanceof Error ? reloadError.message : '模型重新加载失败');
    } finally { setPolicyPreloading(false); }
  }

  async function command(value: 'stop' | 'finish_episode' | 'rerecord_episode' | 'pause_resume' | 'correction' | 'toggle_highlight') {
    pendingSequence.current = event?.sequence ?? 0; setPendingCommand(value);
    try { setRuntime(await postJson<WorkflowRuntime>('/api/runtime/command', { command: value })); }
    catch (commandError) { setPendingCommand(null); setOperationError(commandError instanceof Error ? commandError.message : '操作失败'); }
  }

  const canStart = kind === 'teleoperation'
    || (kind === 'recording' && Boolean(selectedTask))
    || (kind === 'inference' && Boolean(effectivePolicyPath && task.trim()))
    || (kind === 'replay' && Boolean(selectedDataset));
  const canControlEpisode = runningThis && recordingPhase === 'running' && !pendingCommand;
  const canSkipReset = runningThis && recordingPhase === 'resetting' && !pendingCommand;
  const storageRefreshKey = runtimeEvent?.data.stage === 'episode_saved' ? runtimeEvent.sequence : null;
  const residentPolicy = runtime.policy_residency;
  const selectedPolicyResident = residentPolicy?.state === 'ready' && residentPolicy.policy_path === effectivePolicyPath;
  const residentPolicyName = workspace.policies.find((policy) => policy.path === residentPolicy?.policy_path)?.id ?? residentPolicy?.policy_path;
  const policyControls = <PolicyResidencyControls
    resident={residentPolicy}
    residentName={residentPolicyName}
    selectedResident={selectedPolicyResident}
    hasPolicy={Boolean(effectivePolicyPath)}
    runtimeRunning={runtime.running}
    inspecting={policyInspecting}
    preloading={policyPreloading}
    onInspect={inspectPolicy}
    onPreload={preloadPolicy}
    onReload={reloadPolicy}
    onUnload={unloadPolicy}
  />;

  return <section className={`workflow-page${kind === 'teleoperation' || kind === 'recording' ? ' compact-workflow' : ''}${kind === 'teleoperation' ? ' teleoperation-workflow' : ''}${kind === 'recording' ? ' recording-workflow' : ''}${kind === 'inference' ? ' inference-workflow' : ''}`}>
    {statusSlot && createPortal(<WorkflowSummary kind={kind} dataset={selectedDataset} policy={selectedPolicy} policyPath={effectivePolicyPath} event={event} error={visibleError} />, statusSlot)}
    <div className="workflow-grid"><div className="workflow-primary">
      {kind === 'teleoperation' && <WorkflowSection title="控制设置"><div className="form-grid"><label className="full-field">控制频率<select value={fps} onChange={(item) => updateFps(Number(item.target.value))} disabled={runningThis}><option value="30">30 FPS</option><option value="20">20 FPS</option><option value="15">15 FPS</option></select></label></div></WorkflowSection>}
      {kind === 'recording' && <><StorageNotice initial={storage} refreshKey={storageRefreshKey} /><WorkflowSection title="今日采集任务"><div className="form-grid">
        <label className="full-field">任务<select value={taskId} onChange={(item) => { setTaskId(item.target.value); setPolicyInspection(null); }} disabled={runningThis}>{dailyTasks.length === 0 && <option value="">请先在采集进度中创建今日任务</option>}{dailyTasks.map((item) => <option value={item.id} key={item.id}>{item.name} · {item.collection_method === 'policy' ? 'Policy 采集' : '人工采集'}{item.completed ? ' · 已完成' : ''}</option>)}</select></label>
        {selectedTask && <div className="selected-task-description full-field"><span>{selectedTask.collection_method === 'policy' ? 'Policy 采集' : '人工采集'} · 参数由采集进度任务锁定</span><strong>{selectedTask.description}</strong><small>目标 {Math.round(selectedTask.target_duration_s / 60)} 分钟 · 有效 {Math.round(selectedTask.actual_duration_s / 60)} 分钟 · 已保存 {selectedTask.episode_count} Episodes</small></div>}
        {selectedTask?.collection_method === 'policy' && <><label>采集策略<input value={rolloutModes[selectedTask.rollout_strategy].label} readOnly /></label><label>推理后端<input value={selectedTask.inference === 'rtc' ? 'RTC 实时分块' : '同步推理'} readOnly /></label><label className="full-field">本地 Policy<input value={selectedPolicy?.id ?? selectedTask.policy_path} readOnly /></label>{selectedTask.rollout_strategy !== 'episodic_dagger' && <label>最大运行时间<input value={`${selectedTask.duration_s} 秒`} readOnly /></label>}<label>帧率<input value={`${selectedTask.fps} FPS`} readOnly /></label></>}
      </div>{selectedTask?.collection_method === 'policy' && policyControls}{policyInspection && <PolicyInspectionResult inspection={policyInspection} />}</WorkflowSection></>}
      {kind === 'inference' && <WorkflowSection title="本地 Policy 试跑"><div className="form-grid">
        <label className="full-field">Policy<select value={effectivePolicyPath} onChange={(item) => { setPolicyPath(item.target.value); setPolicyInspection(null); }} disabled={runningThis || policyPreloading}>{workspace.policies.length === 0 && <option value="">本机未发现模型</option>}{workspace.policies.map((policy) => <option value={policy.path} key={policy.path}>{policy.id} · {policy.type}</option>)}</select>{effectivePolicyPath && <small className="field-help">本机路径：{effectivePolicyPath}</small>}</label>
        <label>推理后端<select value={inference} onChange={(item) => setInference(item.target.value as RolloutInference)} disabled={runningThis}><option value="sync">同步推理</option><option value="rtc">RTC 实时分块</option></select></label>
        <label>最大运行时间<input type="number" value={duration} onChange={(item) => setDuration(Number(item.target.value))} disabled={runningThis} min="1" /></label>
        <label>帧率<select value={fps} onChange={(item) => setFps(Number(item.target.value))} disabled={runningThis}><option value="30">30 FPS</option><option value="20">20 FPS</option><option value="15">15 FPS</option></select></label>
        <label className="full-field">任务描述<input value={task} onChange={(item) => setTask(item.target.value)} disabled={runningThis} placeholder="使用训练数据中的任务描述效果最稳定" /><small className="field-help">Checkpoint 不记录唯一任务描述；这里是本次推理传给模型的指令。</small></label>
      </div>{policyControls}{policyInspection && <PolicyInspectionResult inspection={policyInspection} />}</WorkflowSection>}
      {kind === 'replay' && <><WorkflowSection title="回放来源"><div className="form-grid"><label className="full-field">数据集<select value={effectiveDatasetId} onChange={(item) => { setDatasetId(item.target.value); setEpisode(0); }} disabled={runningThis}>{workspace.datasets.length === 0 && <option value="">没有本地数据集</option>}{workspace.datasets.map((dataset) => <option value={dataset.id} key={dataset.id}>{dataset.id}</option>)}</select></label><label>Episode<input type="number" value={episode} onChange={(item) => setEpisode(Number(item.target.value))} disabled={runningThis} min="0" max={Math.max(0, (selectedDataset?.episodes ?? 1) - 1)} /></label></div></WorkflowSection><WorkflowSection title="执行设备">{followers.map((follower) => <div className="workflow-device" key={follower.id}><div><strong>{bindingTitle(follower.alias)}</strong><span>{serialIdentity(follower.id)}</span></div><i>已连接</i></div>)}</WorkflowSection></>}
      <div className="workflow-actions">{kind === 'replay' && <span>回放会直接驱动机械臂执行记录动作</span>}<div className={`workflow-command-buttons${runningThis && kind === 'recording' ? ' episode-controls' : ''}`}>
        {runningThis && kind === 'recording' && selectedTask?.collection_method === 'manual' && <><button className="primary" type="button" disabled={!canControlEpisode} onClick={() => void command('finish_episode')}>{pendingCommand === 'finish_episode' && recordingPhase !== 'resetting' ? '正在保存' : '保存这一段'}</button><button className="outline" type="button" disabled={!canControlEpisode} onClick={() => void command('rerecord_episode')}>{pendingCommand === 'rerecord_episode' ? '正在重录' : '重录这一段'}</button>{recordingPhase === 'resetting' && <button className="outline" type="button" disabled={!canSkipReset} onClick={() => void command('finish_episode')}>{pendingCommand === 'finish_episode' ? '正在跳过' : '跳过等待'}</button>}</>}
        {runningThis && kind === 'recording' && (selectedStrategy === 'episodic' || selectedStrategy === 'episodic_dagger') && <><button className="primary" type="button" disabled={Boolean(pendingCommand)} onClick={() => void command('finish_episode')}>{pendingCommand === 'finish_episode' ? '正在切换' : rolloutPhase === 'resetting' ? '跳过重置' : '结束本轮'}</button>{rolloutPhase !== 'resetting' && <button className="outline" type="button" disabled={Boolean(pendingCommand)} onClick={() => void command('rerecord_episode')}>{pendingCommand === 'rerecord_episode' ? '正在重录' : '重录本轮'}</button>}</>}
        {runningThis && kind === 'recording' && rolloutPhase !== 'resetting' && (selectedStrategy === 'dagger_corrections' || selectedStrategy === 'dagger_continuous' || selectedStrategy === 'episodic_dagger') && <>{rolloutPhase === 'autonomous' && <button className="primary" type="button" disabled={Boolean(pendingCommand)} onClick={() => void command('pause_resume')}>暂停 Policy</button>}{rolloutPhase === 'paused' && <><button className="primary" type="button" disabled={Boolean(pendingCommand)} onClick={() => void command('correction')}>开始人工干预</button><button className="outline" type="button" disabled={Boolean(pendingCommand)} onClick={() => void command('pause_resume')}>恢复 Policy</button></>}{rolloutPhase === 'correcting' && <button className="primary" type="button" disabled={Boolean(pendingCommand)} onClick={() => void command('correction')}>结束人工干预</button>}</>}
        {runningThis && kind === 'recording' && selectedStrategy === 'highlight' && <button className="primary" type="button" disabled={Boolean(pendingCommand)} onClick={() => void command('toggle_highlight')}>{rolloutPhase === 'recording' ? '结束片段并保存' : '开始保存片段'}</button>}
        <button className={runningThis ? 'danger' : 'primary'} type="button" disabled={runningOther || Boolean(pendingCommand) || (!runningThis && !canStart)} onClick={() => runningThis ? void command('stop') : void start()}>{runningThis ? pendingCommand === 'stop' ? '正在结束' : kind === 'recording' ? '结束采集' : '停止' : kind === 'teleoperation' ? '开始遥操作' : kind === 'recording' ? '开始采集' : kind === 'inference' ? '开始推理' : '开始回放'}</button>
      </div></div>
    </div></div>
  </section>;
}

function PolicyInspectionResult({ inspection }: { inspection: PolicyInspection }) {
  return <div className={`calibration-progress ${inspection.compatible ? 'done' : 'error'}`}><div><span>{inspection.policy_type.toUpperCase()} · {inspection.revision?.slice(0, 10) ?? '本地模型'}</span><strong>{inspection.compatible ? '模型与当前设备兼容' : '模型与当前设备不兼容'}</strong><small>状态/动作 {inspection.state_dim ?? '—'} / {inspection.action_dim ?? '—'} 维 · 摄像头 {inspection.expected_visuals.length} 路{Object.keys(inspection.rename_map).length > 0 ? ` · 自动映射 ${Object.entries(inspection.rename_map).map(([from, to]) => `${from.split('.').pop()} → ${to.split('.').pop()}`).join(', ')}` : ''}</small>{inspection.issues.map((issue) => <small key={issue}>{issue}</small>)}</div></div>;
}

function PolicyResidencyControls({ resident, residentName, selectedResident, hasPolicy, runtimeRunning, inspecting, preloading, onInspect, onPreload, onReload, onUnload }: {
  resident?: PolicyResidency; residentName?: string; selectedResident: boolean; hasPolicy: boolean;
  runtimeRunning: boolean; inspecting: boolean; preloading: boolean;
  onInspect: () => Promise<void>; onPreload: () => Promise<void>; onReload: () => Promise<void>; onUnload: () => Promise<void>;
}) {
  const state = resident?.state ?? 'empty';
  const loading = state === 'loading' || preloading;
  const ready = state === 'ready';
  const status = loading ? '正在加载到显存' : ready ? selectedResident ? '当前模型已驻留显存' : '显存中驻留了其他模型' : '显存空闲';
  return <div className="policy-residency-controls">
    <div className={`policy-residency ${selectedResident ? 'ready' : state}`}>
      <span>显存模型状态</span><strong>{status}</strong>
      {residentName && state !== 'empty' && <small>{residentName}{resident?.policy_type && resident?.device ? ` · ${resident.policy_type.toUpperCase()} · ${resident.device}` : ''}</small>}
    </div>
    <div className={`policy-actions${ready ? ' resident-ready' : ''}`}>
      <button className="outline" type="button" disabled={runtimeRunning || inspecting || loading || !hasPolicy} onClick={() => void onInspect()}>{inspecting ? '正在检查' : '检查模型兼容性'}</button>
      {ready ? <>
        <button className="outline" type="button" disabled={runtimeRunning || loading || !hasPolicy} onClick={() => void onReload()}>{loading ? '正在重新加载' : selectedResident ? '重新加载' : '加载所选模型'}</button>
        <button className="outline" type="button" disabled={runtimeRunning || loading} onClick={() => void onUnload()}>从显存卸载</button>
      </> : <button className="primary" type="button" disabled={runtimeRunning || loading || !hasPolicy} onClick={() => void onPreload()}>{loading ? '正在加载模型' : '预加载到显存'}</button>}
    </div>
  </div>;
}

function WorkflowSection({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="workflow-section"><h3>{title}</h3>{children}</section>;
}

function WorkflowSummary({ kind, dataset, policy, policyPath, event, error }: { kind: 'teleoperation' | 'recording' | 'inference' | 'replay'; dataset?: LocalDataset; policy?: LocalPolicy; policyPath: string; event: RuntimeEvent | null; error: string }) {
  const failureState = { teleoperation: '遥操作启动失败', recording: '采集失败', inference: '推理失败', replay: '回放失败' }[kind];
  const state = error ? failureState : kind === 'recording' ? rolloutPhaseLabel(event?.data.rollout_phase) ?? event?.message ?? '等待开始' : event?.message || '等待开始';
  const phaseDetail = error && event?.phase !== 'failed' ? '启动失败' : event ? `${event.phase} · ${new Date(event.timestamp).toLocaleTimeString()}` : '尚未启动';
  const errorDetails = error ? <details className="workflow-error-details"><summary>错误详情</summary><pre>{error}</pre></details> : null;
  if (kind === 'teleoperation') return <div className="workflow-summary"><SummaryItem label="运行状态" value={state} detail={event?.data.fps ? `${Number(event.data.fps).toFixed(1)} FPS` : phaseDetail} />{errorDetails}</div>;
  if (kind === 'recording') return <div className="workflow-summary"><SummaryItem label="采集状态" value={state} detail={event?.data.episode !== undefined ? `Episode ${String(event.data.episode)}${event.data.total_episodes !== undefined ? ` / ${String(event.data.total_episodes)}` : ''}` : event?.data.saved_episodes !== undefined ? `已保存 ${String(event.data.saved_episodes)} Episodes` : phaseDetail} />{errorDetails}</div>;
  if (kind === 'inference') return <div className="workflow-summary"><SummaryItem label="运行状态" value={state} detail={typeof event?.data.rollout_phase === 'string' ? String(event.data.rollout_phase) : phaseDetail} /><SummaryItem label="模型" value={policy?.id ?? (policyPath.split('/').slice(-2).join('/') || '未选择')} detail={policy ? `${policy.type} · 本地 checkpoint` : '本机未选择模型'} />{errorDetails}</div>;
  return <div className="workflow-summary"><SummaryItem label="回放状态" value={state} detail={event?.data.frame !== undefined ? `${String(event.data.frame)} / ${String(event.data.total_frames ?? '—')} 帧` : phaseDetail} /><SummaryItem label={dataset ? dataset.id : '数据集'} value={dataset ? `${dataset.frames} 帧` : '未选择'} detail={dataset ? `${dataset.episodes} Episodes · ${dataset.fps || '—'} FPS` : '未发现本地数据集'} />{errorDetails}</div>;
}

function SummaryItem({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div><span>{label}</span><strong>{value}</strong><p>{detail}</p></div>;
}
