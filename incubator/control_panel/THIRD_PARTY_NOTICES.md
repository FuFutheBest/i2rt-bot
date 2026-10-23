# Third-party notices

## Flexiv Rizon 4 / Rizon 4R description assets

The Rizon 4 visual/collision meshes and robot-description material are sourced
from Flexiv Robotics' official `flexiv_ros` repository:

- Source: <https://github.com/flexivrobotics/flexiv_ros>
- Revision: `959774b0adb2d87ae6ca0cfac3e10b4ba18304cf`
- Source directory: `flexiv_description/meshes/rizon4`
- License: Apache License 2.0

The control-panel asset path `static/assets/flexiv` is a symlink to the canonical
repository assets at `assets/robot/rizon`; the standalone Rizon kinematics URDFs
and meshes are stored there and are shared by tools and viewers. The Apache 2.0
license terms apply to these vendored Flexiv description assets.

The right-side Rizon 4R kinematics URDF and complete right-specific mesh set are
from Flexiv Robotics' official `flexiv_rdk` repository:

- Source: <https://github.com/flexivrobotics/flexiv_rdk>
- Revision: `72cc00a137d7dd18816ee5676cc94b630061709b`
- Source directory: `resources`
- License: Apache License 2.0

The viewer loads Rizon 4 and Rizon 4R as independent models with independent
asset maps; the right arm is not synthesized by rotating or reusing the left
model.

## Three.js

The local Three.js runtime, OrbitControls, OBJLoader, and MTLLoader are version
0.160.0 and are distributed under the MIT License. The vendored license text is
at `static/vendor/THREE_LICENSE.txt`. No browser CDN is required.
