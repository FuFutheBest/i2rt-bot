# UMI3 wrist-camera gripper visuals

This directory contains the browser-ready UMI3 gripper visual variants:

- `soft`: long soft fingertips
- `hard`: hard fingertips

Each variant has two links, one revolute joint, and two merged binary STL
meshes. The fixed mesh includes the flange adapter, fixed finger, wrist-camera
mount, and wrist camera. The moving mesh contains the driven finger.

`gripper_joint = 0` is fingertip contact. The upper limit is `pi / 2` radians
and represents a 90-degree opening. The variants have independent closed-pose
origins because their fingertip geometries reach contact at different angles.

The CAD is a physical right-hand assembly. The control-panel viewer mirrors
the complete model across local Y for the left arm and leaves the right arm
unchanged, matching the supplied dual-arm Blender assembly.

The SolidWorks instance matrices were measured after Blender's OBJ import.
Generation converts them back to STEP/CAD axes with
`M_cad = D * M_blender * D`, where `D = diag(-1, -1, 1)`, before merging the
wrist-camera body and upper camera mount into the fixed link. The manifest
records the source matrix, converted matrix, and per-component URDF bounds.

The source STEP archives are user-provided and are not vendored. Regenerate
the meshes with FreeCAD Python:

```powershell
& 'C:\Program Files\FreeCAD 1.0\bin\python.exe' `
  incubator/control_panel/tools/build_umi3_camera_variants.py `
  --hard-dir '<extracted-hard-directory>' `
  --soft-dir '<extracted-soft-directory>'
```

`generation_manifest.json` records source hashes, selected CAD features,
joint-contact measurements, output hashes, triangle counts, and bounds.
