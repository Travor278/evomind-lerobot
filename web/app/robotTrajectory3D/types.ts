export type RobotArmSide = 'left' | 'right';
export type RobotTrajectorySignal = 'action' | 'state';

export type RobotModelManifest = {
  model: string;
  asset_id: string;
  asset_base_url: string;
  urdf_path: string;
  urdf_url: string;
  joint_order: string[];
  feature_mapping: {
    joint_pattern: string;
    gripper_pattern: string | null;
  };
  units: { joint: string; gripper: string };
  gripper: {
    source_range: number[];
    joints: { name: string; scale: number; offset: number }[];
  } | null;
  limits: Record<string, number[]>;
  scene: {
    left_base_xyz: [number, number, number];
    right_base_xyz: [number, number, number];
  };
  files: { path: string; size: number; sha256: string; content_type: string }[];
};

export type RobotTrajectory = {
  time_s: number[];
  frame_count: number;
  arms: Record<RobotArmSide, { joint_values: Record<string, number[]> }>;
};

export type RobotEpisode = {
  robot_type: string | null;
  timestamps: number[];
  series: {
    label: string;
    action_name: string;
    state_name: string;
    action: number[];
    state: number[];
  }[];
};
