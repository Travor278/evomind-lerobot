# SO-101 robot description source

The URDF was generated from the Onshape document linked in
`so101_new_calib.urdf`. Its source STL meshes were merged with their complete
geometry preserved and Meshopt-compressed into `model.glb`; URDF mesh fragments
select the matching link geometry from that single file. Source materials are
omitted because the viewer assigns one runtime material per arm.
