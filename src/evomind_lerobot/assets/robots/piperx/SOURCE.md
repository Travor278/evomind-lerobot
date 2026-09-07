# PiPER-X robot description source

- PiPER-X arm geometry: Evo-RL `src/lerobot/assets/piper_x_description/`
- Evo-RL commit: `081d826f264ce3f641898a844eca2758a679840b`
- PiPER-X source URDF: `piper_x_description_no_gripper.urdf`
- Standard PiPER gripper geometry: AgileX `piper_ros` source below

- Repository: https://github.com/agilexrobotics/piper_ros
- Branch: `noetic`
- Commit: `ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`
- Source paths: `src/piper_description/urdf/piper_description.urdf` and `src/piper_description/meshes/`
- Retrieved: 2026-07-13

Evo-RL's PiPER-X description has no gripper, so the standard AgileX PiPER
gripper is attached to the PiPER-X `link6` flange for visual playback. The
source DAE/STL meshes were merged with their complete geometry preserved and
Meshopt-compressed into `model.glb`; URDF mesh fragments select the matching
link geometry from that single file. Source materials are omitted because the
viewer assigns one runtime material per arm. The upstream MIT license is
preserved in `LICENSE`.
