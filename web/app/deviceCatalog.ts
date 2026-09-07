import { Bot, Hand, Truck, UserRound } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export type DeviceCategoryId = 'arm' | 'hand' | 'humanoid' | 'mobile';
export type ArmMode = 'single' | 'dual';
export type CameraKind = 'wrist' | 'environment';
export type DeviceTransport = 'serial' | 'socketcan';
export type WristSide = 'left' | 'right';
export type CreateStepId = 'category' | 'model' | 'hardware';

export type SystemProfile = {
  id: string;
  label: string;
  robot_type: string;
  teleoperator_type: string | null;
  robot_ports: number | null;
  teleoperator_ports: number | null;
  robot_transport: DeviceTransport | null;
  teleoperator_transport: DeviceTransport | null;
};

export type DeviceCategory = {
  id: DeviceCategoryId;
  title: string;
  icon: LucideIcon;
};

export type DeviceVariant = {
  mode: ArmMode;
  profile: SystemProfile;
};

export type DeviceModelOption = {
  id: string;
  title: string;
  category: DeviceCategoryId;
  description: string;
  image?: string;
  imageAlt?: string;
  variants: DeviceVariant[];
};

export type CameraDraft = {
  id: string;
  kind: CameraKind;
  side?: WristSide;
  label?: string;
};

let cameraDraftSequence = 0;

function newCameraDraftId() {
  cameraDraftSequence += 1;
  return `camera-${Date.now().toString(36)}-${cameraDraftSequence}`;
}

export const categories: DeviceCategory[] = [
  { id: 'arm', title: '机械臂', icon: Bot },
  { id: 'hand', title: '灵巧手', icon: Hand },
  { id: 'humanoid', title: '人形机器人', icon: UserRound },
  { id: 'mobile', title: '移动机器人', icon: Truck },
];

export const createSteps: { id: CreateStepId; label: string }[] = [
  { id: 'category', label: '选择类型' },
  { id: 'model', label: '选择型号' },
  { id: 'hardware', label: '本体配置' },
];

const modelDefinitions: Omit<DeviceModelOption, 'variants'>[] = [
  { id: 'so101', title: 'SO-101', category: 'arm', description: '桌面机械臂', image: '/devices/so101.webp', imageAlt: 'SO-101 机械臂' },
  {
    id: 'piperx',
    title: 'PiperX',
    category: 'arm',
    description: 'SocketCAN 工业机械臂',
    image: '/devices/piperx-card.webp',
    imageAlt: 'AgileX PiPER-X 工业机械臂',
  },
  { id: 'openarm', title: 'OpenArm', category: 'arm', description: '开源机械臂', image: '/devices/openarm.png', imageAlt: 'OpenArm 机械臂' },
  { id: 'rebot', title: 'reBot', category: 'arm', description: '桌面机械臂', image: '/devices/rebot.jpg', imageAlt: 'reBot B601-DM 机械臂' },
  { id: 'omx', title: 'OMX', category: 'arm', description: '桌面机械臂', image: '/devices/omx.webp', imageAlt: 'OMX 机械臂' },
  { id: 'koch', title: 'Koch', category: 'arm', description: '桌面机械臂', image: '/devices/koch.jpg', imageAlt: 'Koch 机械臂' },
  { id: 'hope_jr_hand', title: 'Hope Jr Hand', category: 'hand', description: '灵巧手', image: '/devices/hope-jr.png', imageAlt: 'Hope Jr 灵巧手' },
  { id: 'reachy2', title: 'Reachy 2', category: 'humanoid', description: '人形机器人', image: '/devices/reachy2.jpg', imageAlt: 'Reachy 2 人形机器人' },
  { id: 'unitree_g1', title: 'Unitree G1', category: 'humanoid', description: '人形机器人', image: '/devices/unitree-g1.jpg', imageAlt: 'Unitree G1 人形机器人' },
  { id: 'earthrover_mini_plus', title: 'EarthRover Mini Plus', category: 'mobile', description: '移动机器人', image: '/devices/earthrover.webp', imageAlt: 'EarthRover Mini Plus 移动机器人' },
  { id: 'lekiwi', title: 'LeKiwi', category: 'mobile', description: '移动操作机器人', image: '/devices/lekiwi.jpg', imageAlt: 'LeKiwi 移动操作机器人' },
  { id: 'lekiwi_client', title: 'LeKiwi Client', category: 'mobile', description: '移动操作机器人客户端', image: '/devices/lekiwi.jpg', imageAlt: 'LeKiwi 移动操作机器人' },
];

const groupedProfiles: Record<string, Partial<Record<ArmMode, string>>> = {
  so101: { single: 'so101', dual: 'bi_so' },
  piperx: { single: 'piperx', dual: 'bi_piperx' },
  openarm: { single: 'openarm', dual: 'bi_openarm' },
  rebot: { single: 'rebot', dual: 'bi_rebot' },
  omx: { single: 'omx' },
  koch: { single: 'koch' },
  hope_jr_hand: { single: 'hope_jr_hand' },
  reachy2: { single: 'reachy2' },
  unitree_g1: { single: 'unitree_g1' },
  earthrover_mini_plus: { single: 'earthrover_mini_plus' },
  lekiwi: { single: 'lekiwi' },
  lekiwi_client: { single: 'lekiwi_client' },
};

const hiddenProfileIds = new Set(['so100', 'hope_jr_arm']);

function profileCategory(profile: SystemProfile): DeviceCategoryId {
  if (profile.id.includes('hand')) return 'hand';
  if (profile.id === 'reachy2' || profile.id.includes('unitree')) return 'humanoid';
  if (profile.id.includes('earthrover') || profile.id.includes('lekiwi')) return 'mobile';
  return 'arm';
}

function profileMode(profile: SystemProfile): ArmMode {
  return profile.robot_ports === 2 || profile.teleoperator_ports === 2 ? 'dual' : 'single';
}

export function modelsFromProfiles(profiles: SystemProfile[]): DeviceModelOption[] {
  const byId = new Map(profiles.map((profile) => [profile.id, profile]));
  const consumed = new Set<string>();
  const models: DeviceModelOption[] = [];

  for (const definition of modelDefinitions) {
    const variants = Object.entries(groupedProfiles[definition.id]).flatMap(([mode, profileId]) => {
      const profile = profileId ? byId.get(profileId) : undefined;
      if (!profile) return [];
      consumed.add(profile.id);
      return [{ mode: mode as ArmMode, profile }];
    });
    if (variants.length > 0) models.push({ ...definition, variants });
  }

  for (const profile of profiles) {
    if (consumed.has(profile.id) || hiddenProfileIds.has(profile.id)) continue;
    models.push({
      id: profile.id,
      title: profile.label,
      category: profileCategory(profile),
      description: profile.robot_type,
      variants: [{ mode: profileMode(profile), profile }],
    });
  }
  return models;
}

export function defaultCameras(mode: ArmMode, category: DeviceCategoryId = 'arm'): CameraDraft[] {
  if (category === 'mobile') {
    return [
      { id: newCameraDraftId(), kind: 'environment', label: '环境摄像头 1' },
      { id: newCameraDraftId(), kind: 'environment', label: '环境摄像头 2' },
    ];
  }
  if (mode === 'dual') {
    return [
      { id: newCameraDraftId(), kind: 'wrist', side: 'left' },
      { id: newCameraDraftId(), kind: 'wrist', side: 'right' },
      { id: newCameraDraftId(), kind: 'environment', label: '环境摄像头 1' },
    ];
  }
  return [
    { id: newCameraDraftId(), kind: 'wrist', label: '腕部摄像头' },
    { id: newCameraDraftId(), kind: 'environment', label: '环境摄像头 1' },
  ];
}

export function armModeLabel(mode: ArmMode) {
  return mode === 'dual' ? '双臂' : '单臂';
}

export function cameraKindLabel(kind: CameraKind) {
  return kind === 'wrist' ? '腕部摄像头' : '环境摄像头';
}

export function cameraDisplayLabel(camera: CameraDraft, cameras: CameraDraft[], index: number, mode: ArmMode) {
  if (camera.kind === 'wrist') {
    if (mode === 'dual') return camera.side === 'right' ? '右腕摄像头' : '左腕摄像头';
    return camera.label?.trim() || '腕部摄像头';
  }
  const environmentIndex = cameras.slice(0, index + 1).filter((item) => item.kind === 'environment').length;
  return camera.label?.trim() || `环境摄像头 ${environmentIndex}`;
}

export function cameraSlotId(camera: CameraDraft, cameras: CameraDraft[], index: number, mode: ArmMode) {
  if (camera.kind === 'wrist') {
    if (mode === 'dual') return `${camera.side ?? 'left'}_wrist`;
    return 'wrist';
  }
  const environmentIndex = cameras.slice(0, index + 1).filter((item) => item.kind === 'environment').length;
  return `environment_${environmentIndex}`;
}

export function nextCameraDraft(cameras: CameraDraft[], mode: ArmMode, kind: CameraKind): CameraDraft | null {
  if (kind === 'wrist') {
    const limit = mode === 'dual' ? 2 : 1;
    const wristCameras = cameras.filter((camera) => camera.kind === 'wrist');
    if (wristCameras.length >= limit) return null;
    const side = mode === 'dual' && wristCameras.some((camera) => camera.side === 'left') ? 'right' : 'left';
    return { id: newCameraDraftId(), kind, side: mode === 'dual' ? side : undefined };
  }
  if (cameras.filter((camera) => camera.kind === 'environment').length >= 2) return null;
  return { id: newCameraDraftId(), kind };
}

export function canAddCameraKind(cameras: CameraDraft[], mode: ArmMode, kind: CameraKind) {
  const limit = kind === 'environment' ? 2 : mode === 'dual' ? 2 : 1;
  return cameras.filter((camera) => camera.kind === kind).length < limit;
}
