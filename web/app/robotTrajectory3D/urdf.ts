import * as THREE from 'three';
import { MeshoptDecoder } from 'meshoptimizer';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import URDFLoader, { type URDFRobot } from 'urdf-loader';

export async function loadRobotUrdf(urdfUrl: string, geometryUrl: string): Promise<URDFRobot> {
  const geometry = await loadGeometryLibrary(geometryUrl);
  const loader = new URDFLoader();
  loader.loadMeshCb = (path, _manager, done) => {
    const source = geometry.get(geometryKey(path));
    if (!source) throw new Error(`3D 几何资源缺失：${path}`);
    done(new THREE.Group().add(source.clone(true)));
  };
  return loader.loadAsync(urdfUrl);
}

async function loadGeometryLibrary(url: string): Promise<Map<string, THREE.Object3D>> {
  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  const gltf = await loader.loadAsync(url);
  const geometry = new Map<string, THREE.Object3D>();
  gltf.scene.traverse((object) => { if (object.name) geometry.set(object.name, object); });
  return geometry;
}

function geometryKey(url: string): string {
  const fragmentIndex = url.indexOf('#');
  if (fragmentIndex < 0 || fragmentIndex === url.length - 1) throw new Error(`URDF mesh 缺少几何标识：${url}`);
  return THREE.PropertyBinding.sanitizeNodeName(decodeURIComponent(url.slice(fragmentIndex + 1)));
}
