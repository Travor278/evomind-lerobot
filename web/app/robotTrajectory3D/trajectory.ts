import type { RobotArmSide, RobotEpisode, RobotModelManifest, RobotTrajectory, RobotTrajectorySignal } from './types';

const DEG_TO_RAD = Math.PI / 180;
const ARM_SIDES: RobotArmSide[] = ['left', 'right'];

export function buildRobotTrajectory(
  episode: RobotEpisode,
  signal: RobotTrajectorySignal,
  manifest: RobotModelManifest,
): { trajectory: RobotTrajectory | null; error: string } {
  const valuesByFeature = new Map<string, number[]>();
  for (const item of episode.series) {
    const name = signal === 'action' ? item.action_name : item.state_name;
    const values = signal === 'action' ? item.action : item.state;
    if (name && values.length) valuesByFeature.set(normalizeFeature(name), values);
  }
  if (!episode.timestamps.length || !valuesByFeature.size) {
    return { trajectory: null, error: `${signal === 'action' ? 'Action' : 'Observation'} 没有可用的关节序列` };
  }

  const arms: RobotTrajectory['arms'] = {
    left: { joint_values: {} },
    right: { joint_values: {} },
  };
  for (const side of ARM_SIDES) {
    for (const [jointIndex, joint] of manifest.joint_order.entries()) {
      const values = findFeature(valuesByFeature, jointCandidates(manifest.model, side, joint, jointIndex + 1));
      if (!values) continue;
      arms[side].joint_values[joint] = convertJointSeries(values, joint, manifest);
    }
    if (manifest.gripper) {
      const values = findFeature(valuesByFeature, gripperCandidates(side));
      if (values) {
        const [sourceMin, sourceMax] = manifest.gripper.source_range;
        for (const joint of manifest.gripper.joints) {
          arms[side].joint_values[joint.name] = values.map((value) => (
            clamp(value, sourceMin, sourceMax) * joint.scale + joint.offset
          ));
        }
      }
    }
  }

  if (!ARM_SIDES.some((side) => Object.keys(arms[side].joint_values).length > 0)) {
    return { trajectory: null, error: '数据中的关节名称与此 3D 模型不匹配' };
  }
  return {
    error: '',
    trajectory: { time_s: episode.timestamps, frame_count: episode.timestamps.length, arms },
  };
}

function jointCandidates(model: string, side: RobotArmSide, joint: string, index: number): string[] {
  const sided = model === 'piperx'
    ? [`${side}_joint_${index}.pos`, `${side}_joint${index}.pos`]
    : [`${side}_${joint}.pos`];
  if (side === 'right') return sided;
  return model === 'piperx'
    ? [...sided, `joint_${index}.pos`, `joint${index}.pos`]
    : [...sided, `${joint}.pos`];
}

function gripperCandidates(side: RobotArmSide): string[] {
  const sided = [`${side}_gripper.pos`, `${side}_gripper`];
  return side === 'left' ? [...sided, 'gripper.pos', 'gripper'] : sided;
}

function findFeature(features: Map<string, number[]>, candidates: string[]): number[] | undefined {
  for (const candidate of candidates) {
    const values = features.get(normalizeFeature(candidate));
    if (values) return values;
  }
  return undefined;
}

function normalizeFeature(value: string): string {
  return value.toLowerCase().replace(/\.pos$/u, '').replace(/[^a-z0-9]+/gu, '_').replace(/^_+|_+$/gu, '');
}

function convertJointSeries(values: number[], joint: string, manifest: RobotModelManifest): number[] {
  const limits = manifest.limits[joint];
  return values.map((value) => {
    const degrees = manifest.units.joint === 'degree' ? value : value * 180 / Math.PI;
    return (limits ? clamp(degrees, limits[0], limits[1]) : degrees) * DEG_TO_RAD;
  });
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
