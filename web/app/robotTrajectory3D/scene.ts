import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { URDFRobot } from 'urdf-loader';
import { loadRobotUrdf } from './urdf';
import type { RobotArmSide, RobotModelManifest, RobotTrajectory } from './types';

const ARM_COLORS: Record<RobotArmSide, string> = { left: '#4f8fbd', right: '#d0892b' };
const CAMERA_VIEW_DIRECTION = new THREE.Vector3(-1.35, -1.1, 0.85).normalize();

export class RobotTrajectory3DScene {
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly resizeObserver: ResizeObserver;
  private readonly arms = new Map<RobotArmSide, { robot: URDFRobot; root: THREE.Group }>();
  private disposed = false;

  constructor(private readonly container: HTMLDivElement) {
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 420;
    this.scene.background = new THREE.Color(0xf3f6fa);
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.04, 40);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-1.15, -0.9, 0.72);
    this.camera.lookAt(0.18, 0, 0.22);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setSize(width, height);
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0.18, 0, 0.22);
    this.controls.addEventListener('change', this.render);
    this.controls.update();
    this.scene.add(buildLights(), buildFloor(), new THREE.AxesHelper(0.12));
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.render();
  }

  async loadArms(model: RobotModelManifest, sides: RobotArmSide[]): Promise<void> {
    if (this.disposed || sides.length === 0) return;
    const prototype = await loadRobotUrdf(model.urdf_url, robotGeometryUrl(model));
    if (this.disposed) { disposeObject(prototype); return; }
    const materials = collectMaterials(prototype);
    sides.forEach((side, index) => {
      const robot = index === 0 ? prototype : prototype.clone(true) as URDFRobot;
      const root = new THREE.Group();
      root.position.fromArray(sides.length === 1 ? [0, 0, 0] : side === 'left' ? model.scene.left_base_xyz : model.scene.right_base_xyz);
      applyArmMaterial(robot, ARM_COLORS[side]);
      root.add(robot);
      this.scene.add(root);
      this.arms.set(side, { robot, root });
    });
    materials.forEach((material) => material.dispose());
    this.frameArms();
  }

  applyTime(trajectory: RobotTrajectory, model: RobotModelManifest, currentTime: number): void {
    const frame = frameForTime(trajectory.time_s, currentTime);
    for (const side of ['left', 'right'] as const) {
      const arm = this.arms.get(side);
      if (!arm) continue;
      const series = trajectory.arms[side].joint_values;
      const joints = [...model.joint_order, ...(model.gripper?.joints.map((joint) => joint.name) ?? [])];
      for (const joint of joints) {
        const values = series[joint];
        if (!values?.length) continue;
        const value = values[frame.index] + (values[frame.nextIndex] - values[frame.index]) * frame.alpha;
        arm.robot.setJointValue(joint, value);
      }
      arm.robot.updateMatrixWorld(true);
    }
    this.render();
  }

  dispose(): void {
    this.disposed = true;
    this.resizeObserver.disconnect();
    this.controls.removeEventListener('change', this.render);
    this.controls.dispose();
    disposeObject(this.scene);
    this.scene.clear();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.renderer.domElement.remove();
  }

  private frameArms(): void {
    const bounds = new THREE.Box3();
    this.arms.forEach((arm) => { arm.root.updateMatrixWorld(true); bounds.expandByObject(arm.root, true); });
    if (bounds.isEmpty()) return;
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(Math.hypot(size.x, size.y, size.z) / 2, 0.05);
    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * Math.max(this.camera.aspect, 0.01));
    const distance = radius / Math.sin(Math.max(Math.min(verticalFov, horizontalFov), THREE.MathUtils.degToRad(5)) / 2) * 1.25;
    this.camera.position.copy(center).addScaledVector(CAMERA_VIEW_DIRECTION, distance);
    this.camera.near = Math.max(0.02, distance / 100);
    this.camera.far = Math.max(40, distance * 12);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.maxDistance = distance * 4;
    this.controls.update();
    this.render();
  }

  private resize(): void {
    const { clientWidth: width, clientHeight: height } = this.container;
    if (width <= 0 || height <= 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.render();
  }

  private readonly render = () => { this.renderer.render(this.scene, this.camera); };
}

function frameForTime(times: number[], currentTime: number) {
  if (!times.length || currentTime <= times[0]) return { index: 0, nextIndex: 0, alpha: 0 };
  const lastIndex = times.length - 1;
  if (currentTime >= times[lastIndex]) return { index: lastIndex, nextIndex: lastIndex, alpha: 0 };
  let low = 1;
  let high = lastIndex;
  while (low < high) {
    const midpoint = Math.floor((low + high) / 2);
    if (currentTime > times[midpoint]) low = midpoint + 1; else high = midpoint;
  }
  const span = Math.max(times[low] - times[low - 1], Number.EPSILON);
  return { index: low - 1, nextIndex: low, alpha: Math.min(1, Math.max(0, (currentTime - times[low - 1]) / span)) };
}

function buildLights(): THREE.Group {
  const group = new THREE.Group();
  const key = new THREE.DirectionalLight(0xffffff, 0.78);
  const fill = new THREE.DirectionalLight(0xffffff, 0.38);
  key.position.set(0.8, 0.7, 1.6);
  fill.position.set(-0.6, -0.5, 0.7);
  group.add(new THREE.AmbientLight(0xffffff, 0.62), key, fill);
  return group;
}

function buildFloor(): THREE.Group {
  const group = new THREE.Group();
  const size = 1.5;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshStandardMaterial({ color: 0xe6ebf2, roughness: 0.92 }));
  floor.position.z = -0.001;
  group.add(floor, new THREE.GridHelper(size, 30, 0x94a3b8, 0xcbd5e1).rotateX(Math.PI / 2));
  return group;
}

function robotGeometryUrl(model: RobotModelManifest): string {
  const geometry = model.files.find((file) => file.path === 'model.glb');
  if (!geometry) throw new Error('3D 模型缺少 model.glb');
  return new URL(`model.glb?v=${geometry.sha256}`, new URL(model.asset_base_url, window.location.origin)).toString();
}

function applyArmMaterial(robot: URDFRobot, color: string): void {
  const material = new THREE.MeshStandardMaterial({ color, metalness: 0.18, roughness: 0.72 });
  robot.traverse((child) => { if (child instanceof THREE.Mesh) child.material = material; });
}

function collectMaterials(object: THREE.Object3D): Set<THREE.Material> {
  const result = new Set<THREE.Material>();
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((material) => result.add(material));
  });
  return result;
}

function disposeObject(object: THREE.Object3D): void {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  object.traverse((child) => {
    const item = child as THREE.Object3D & { geometry?: THREE.BufferGeometry; material?: THREE.Material | THREE.Material[] };
    if (item.geometry) geometries.add(item.geometry);
    (Array.isArray(item.material) ? item.material : item.material ? [item.material] : []).forEach((material) => materials.add(material));
  });
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach((material) => material.dispose());
}
