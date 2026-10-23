from pathlib import Path
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

from flask import Flask


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "static" / "assets" / "flexiv"
UMI3_ASSET_ROOT = ROOT / "static" / "assets" / "tools" / "umi3_camera"
YAM_ASSET_ROOT = ROOT / "static" / "assets" / "yam"
YAM_MESH_SHA256 = {
    "base.stl": "d95a5aa0c06a5ef6da02788b0df5a14e4202c1ae8eb2771a7f6f8947e3a490c7",
    "link1.stl": "41f7dfbd0471a0f96884c73782e82927f2bbc6aced9b55c0111da1b52d6e0c50",
    "link2.stl": "067b51567a95a631d50fc02515bd045a2eb7f2f34489d5aec31d9f3f6ce58b30",
    "link3.stl": "9650a114393881b54a68822ba3fb48fcd3318023ebecaf251e4f7804e91f801c",
    "link4.stl": "42740ce2bb8693760a35dc6e9e8fe0981999c02d6bf167a465c9936acf41c557",
    "link5.stl": "f70eb1191d6a1ccbb748a46fe8ba541f38094c9829fd1859300e70360c5a07c7",
}
UMI3_ASSET_SHA256 = {
    "hard/umi3_gripper.urdf": "7eb6b6cf2193fd9acec5434341384d5b4813f9cb7cf347178d59ca32a2152d74",
    "hard/meshes/base_link.stl": "249baaa819795332bcfe7e27ec89e586ba87df9803c80d8f86a3d5e9e9b2ca14",
    "hard/meshes/index_link.stl": "01d5105e8f851532730bdd9ce57962f1210dbc2d4ca7a8ebfda2bf19af9a46bd",
    "soft/umi3_gripper.urdf": "007fa46e99881d7ee6feaa318b8a4d47c28ace04625e2e7472401b5da975093e",
    "soft/meshes/base_link.stl": "3277d8a0017aba685f4d109b0db020ae5d0dc5d13518c7567c248437a43cd142",
    "soft/meshes/index_link.stl": "41a025b0d5cb7777519a674a377619936201252efd41a985804a04692fba463c",
}
R4R_URDF_SHA256 = "40efa19fc4414179fd0e598a15f2901dc7ad9b14c3620598c3e380caa4766fb4"
R4R_MESH_MANIFEST_SHA256 = "9034fbc1efb5911e7c90ec66bbbaeb007be12d37eca5c5103056576001b0af04"
R4R_KEY_MESH_SHA256 = {
    "link3r.obj": "1657f48615834abaeeb9295ec60454b02beb2e6c13905b3f5204439b5ef1a37f",
    "link4r.obj": "f80f661ca72c1429e65d391e4b520aaddf4d2cf639d4fb9371a5048c025ee1f2",
    "link6r.obj": "b38000aafa82b6f7039888551bdb9fcc3bf03f6a41745fd95da9fde4f8902952",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256sum_manifest(directory: Path) -> str:
    rows = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = "./" + path.relative_to(directory).as_posix()
        rows.append(f"{_sha256(path)}  {relative}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _bounds_overlap(first: dict, second: dict) -> bool:
    return all(
        first["min"][axis] <= second["max"][axis]
        and second["min"][axis] <= first["max"][axis]
        for axis in range(3)
    )


def test_umi3_camera_gripper_variants_and_binary_stls_are_vendored_verbatim():
    for relative, expected_sha256 in UMI3_ASSET_SHA256.items():
        path = UMI3_ASSET_ROOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == expected_sha256
    assert (UMI3_ASSET_ROOT / "README.md").is_file()
    manifest = json.loads(
        (UMI3_ASSET_ROOT / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "superinference.umi3-camera-visual.v2"
    assert "C:\\Users" not in json.dumps(manifest)
    assert manifest["coordinate_conversion"]["blender_obj_from_cad_axis"] == [
        -1.0,
        -1.0,
        1.0,
    ]

    closed_origins = {}
    for variant in ("soft", "hard"):
        root = ET.parse(UMI3_ASSET_ROOT / variant / "umi3_gripper.urdf").getroot()
        joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
        joint = joints["gripper_joint"]
        assert joint.attrib["type"] == "revolute"
        assert joint.find("axis").attrib["xyz"] == "0 -1 0"
        assert float(joint.find("limit").attrib["lower"]) == 0.0
        assert math.isclose(float(joint.find("limit").attrib["upper"]), math.pi / 2)
        closed_origins[variant] = float(joint.find("origin").attrib["rpy"].split()[1])
        meshes = {mesh.attrib["filename"] for mesh in root.findall(".//mesh")}
        assert meshes == {"meshes/base_link.stl", "meshes/index_link.stl"}
        joint_report = manifest["variants"][variant]["joint"]
        assert joint_report["closed_contact_distance_mm"] <= 1e-5
        assert joint_report["open_contact_distance_mm"] > 90.0
        transforms = manifest["variants"][variant]["component_transforms"]
        assert transforms["camera"]["cad"] == [
            [0.0, -1.0, 0.0, 0.08627204748814458],
            [0.0, 0.0, 1.0, 0.004306474806139823],
            [-1.0, 0.0, 0.0, -0.01254095684420557],
            [0.0, 0.0, 0.0, 1.0],
        ]
        assert transforms["camera_mount"]["cad"] == [
            [0.0, -1.0, 0.0, 0.06531450287586101],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        component_bounds = manifest["variants"][variant]["component_bounds_urdf_mm"]
        body_bounds = component_bounds["body"]
        camera_bounds = component_bounds["camera"]
        mount_bounds = component_bounds["camera_mount"]
        assert 10.0 < camera_bounds["min"][2] < 80.0
        assert 10.0 < camera_bounds["max"][2] < 90.0
        assert 20.0 < mount_bounds["min"][2] < 90.0
        assert 20.0 < mount_bounds["max"][2] < 90.0
        assert _bounds_overlap(body_bounds, mount_bounds)
        assert _bounds_overlap(mount_bounds, camera_bounds)
    assert not math.isclose(closed_origins["soft"], closed_origins["hard"], abs_tol=0.1)


def test_flexiv_left_and_right_urdfs_and_meshes_are_vendored():
    expected_names = {"Rizon4": "Rizon4", "Rizon4R": "Rizon4R"}
    for directory_name, robot_name in expected_names.items():
        urdf = ASSET_ROOT / f"flexiv_{directory_name}_kinematics.urdf"
        assert urdf.is_file(), urdf
        root = ET.parse(urdf).getroot()
        assert root.attrib["name"] == robot_name
        meshes = [
            mesh.attrib["filename"].replace("package://", "")
            for mesh in root.findall(".//geometry/mesh")
            if "filename" in mesh.attrib
        ]
        assert len(meshes) >= 16
        for relative in meshes:
            path = ASSET_ROOT / relative
            assert path.is_file(), path
            assert path.stat().st_size > 1000, path

            if "/visual/" in relative and path.suffix == ".obj":
                material = path.with_suffix(".mtl")
                assert material.is_file(), material


def test_yam_urdf_uses_six_commanded_joints_and_vendored_i2rt_meshes():
    urdf = YAM_ASSET_ROOT / 'yam_no_gripper.urdf'
    assert urdf.is_file()
    root = ET.parse(urdf).getroot()
    joints = root.findall('joint')
    assert joints[0].attrib == {'name': 'yam_model_alignment', 'type': 'fixed'}
    commanded = [joint for joint in joints if joint.attrib['type'] != 'fixed']
    assert [joint.attrib['name'] for joint in commanded] == [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
    ]
    assert root.find("joint[@name='joint6']/child").attrib['link'] == 'end_effector'
    meshes = {mesh.attrib['filename'] for mesh in root.findall('.//mesh')}
    assert meshes == {f'meshes/{name}' for name in YAM_MESH_SHA256}
    for filename, expected_hash in YAM_MESH_SHA256.items():
        path = YAM_ASSET_ROOT / 'meshes' / filename
        assert path.is_file()
        assert _sha256(path) == expected_hash

    app = Flask(
        'control-panel-yam-static-contract',
        static_folder=str(ROOT / 'static'),
        static_url_path='/static',
    )
    client = app.test_client()
    for relative in ['yam_no_gripper.urdf', *[f'meshes/{name}' for name in YAM_MESH_SHA256]]:
        response = client.get(f'/static/assets/yam/{relative}')
        assert response.status_code == 200, relative
        assert response.data, relative


def test_rizon4r_assets_match_the_approved_vendored_snapshot():
    urdf = ASSET_ROOT / "flexiv_Rizon4R_kinematics.urdf"
    mesh_root = ASSET_ROOT / "meshes" / "Rizon4R"
    assert _sha256(urdf) == R4R_URDF_SHA256
    assert _sha256sum_manifest(mesh_root) == R4R_MESH_MANIFEST_SHA256
    for filename, expected_hash in R4R_KEY_MESH_SHA256.items():
        assert _sha256(mesh_root / "visual" / filename) == expected_hash


def test_rizon4r_static_urls_are_served_with_the_right_specific_meshes():
    app = Flask(
        "control-panel-static-contract",
        static_folder=str(ROOT / "static"),
        static_url_path="/static",
    )
    client = app.test_client()
    urls = set()
    for variant in ("Rizon4", "Rizon4R"):
        urdf_path = ASSET_ROOT / f"flexiv_{variant}_kinematics.urdf"
        urls.add(f"/static/assets/flexiv/{urdf_path.name}")
        for mesh in ET.parse(urdf_path).getroot().findall(".//geometry/mesh"):
            package_path = mesh.attrib.get("filename", "")
            assert package_path.startswith("package://meshes/"), package_path
            relative = package_path.removeprefix("package://")
            urls.add(f"/static/assets/flexiv/{relative}")
            if "/visual/" in relative and relative.endswith(".obj"):
                urls.add(f"/static/assets/flexiv/{relative[:-4]}.mtl")

    for url in urls:
        response = client.get(url)
        assert response.status_code == 200, url
        assert response.data, url


def test_viewer_is_local_and_exposes_joint_update_api():
    viewer = ROOT / "static" / "arm_viewer.js"
    source = viewer.read_text(encoding="utf-8")
    assert "OrbitControls" in source
    assert "updateJointStates(left, right)" in source
    assert "new THREE.WebGLRenderer" in source
    assert "https://" not in source
    assert re.search(r"#arm-viewer-root", source) is not None
    for dependency in (
        ROOT / "static" / "vendor" / "three.module.min.js",
        ROOT / "static" / "vendor" / "OrbitControls.js",
        ROOT / "static" / "vendor" / "OBJLoader.js",
        ROOT / "static" / "vendor" / "MTLLoader.js",
    ):
        assert dependency.is_file(), dependency


def test_viewer_adapts_flexiv_z_up_to_three_y_up():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert "ARM_BASE_Y_OFFSET = 0.2" in source
    assert "left: Object.freeze([0, ARM_BASE_Y_OFFSET, 0])" in source
    assert "right: Object.freeze([0, -ARM_BASE_Y_OFFSET, 0])" in source
    assert "left: ARM_BASE_Y_OFFSET" in source
    assert "right: -ARM_BASE_Y_OFFSET" in source
    assert ".makeRotationZ(Math.PI)" in source
    assert ".setPosition(0, yOffset, 0)" in source
    assert "stationFromRobot.multiply(robotFromTcp)" in source
    assert "flexivWorldToThreePosition(ARM_BASE_POSITIONS.left)" in source
    assert "flexivWorldToThreePosition(ARM_BASE_POSITIONS.right)" in source
    assert "return new THREE.Vector3(x, z, -y)" in source
    assert ".multiplyQuaternions(rotateWorldYSecond, rotateWorldZFirst)" in source
    assert ".multiply(rotation)" in source
    assert ".multiply(FLEXIV_TO_THREE_QUATERNION.clone().invert())" in source
    assert "armRoot.quaternion.copy(armBaseSceneQuaternion())" in source
    assert "urdfFrame.quaternion.copy(FLEXIV_TO_THREE_QUATERNION)" in source
    assert "this.worldFrame.quaternion.copy(FLEXIV_TO_THREE_QUATERNION)" in source
    assert "identity.marker.quaternion.copy(armRoot.quaternion).invert()" not in source
    assert "ARM_BASE_YAWS" not in source
    assert "armRoot.rotation" not in source
    assert "GROUND_FLEXIV_POSITION = Object.freeze([0, 0, -0.5])" in source
    assert "return flexivWorldToThreePosition(GROUND_FLEXIV_POSITION)" in source
    assert "this.floor.rotation.x = -Math.PI / 2" in source
    assert "this.floor.position.copy(defaultViewTarget)" in source
    assert "this.grid.position.copy(defaultViewTarget)" in source
    assert "this.grid.position.y += GRID_SCENE_LIFT" in source
    assert "this.worldFrame.position.set(0, 0, 0)" in source
    assert source.count("this.controls.target.copy(defaultViewTarget)") == 2
    assert "this.controls.target.copy(groundScenePosition())" in source
    assert source.count("this.camera.lookAt(defaultViewTarget)") == 2
    assert "this.camera.lookAt(this.controls.target)" in source
    assert source.count("this.controls.target.copy(center)") == 1
    assert source.count("this.camera.lookAt(center)") == 1
    assert "_frameRecordedTcpTrajectories(points)" in source
    assert "this.controls.target.set(0, 0.65, 0)" not in source
    assert "DEFAULT_VIEW_PADDING = 1.22" in source
    assert "horizontalFov" in source
    assert "Math.max(2.25, fitHeight, fitWidth) + size.z * 0.5" in source
    assert "center.x + distance * 0.38" in source
    assert "center.z + distance * 0.92" in source


def test_base_pose_quaternion_order_and_flexiv_to_three_basis_mapping():
    def multiply(left, right):
        lx, ly, lz, lw = left
        rx, ry, rz, rw = right
        return (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )

    def inverse(quaternion):
        x, y, z, w = quaternion
        return (-x, -y, -z, w)

    def rotate(quaternion, vector):
        pure = (vector[0], vector[1], vector[2], 0.0)
        result = multiply(multiply(quaternion, pure), inverse(quaternion))
        return result[:3]

    def close(actual, expected):
        return all(abs(a - e) < 1e-9 for a, e in zip(actual, expected))

    def map_position(position):
        x, y, z = position
        return (x, z, -y)

    assert close(map_position((0, 0.2, 0)), (0, 0, -0.2))
    assert close(map_position((0, -0.2, 0)), (0, 0, 0.2))
    assert close(map_position((0, 0, -0.5)), (0, -0.5, 0))

    half = math.sqrt(0.5)
    rotate_world_z_first = (0.0, 0.0, 1.0, 0.0)
    rotate_world_y_second = (0.0, half, 0.0, half)
    # Three.js a.multiply(b) stores a * b; b acts first on a vector.
    flexiv_base = multiply(rotate_world_y_second, rotate_world_z_first)
    assert close(flexiv_base, (half, 0.0, half, 0.0))
    assert close(rotate(flexiv_base, (1, 0, 0)), (0, 0, 1))
    assert close(rotate(flexiv_base, (0, 1, 0)), (0, -1, 0))
    assert close(rotate(flexiv_base, (0, 0, 1)), (1, 0, 0))

    flexiv_to_three = (-half, 0.0, 0.0, half)
    scene_base = multiply(
        multiply(flexiv_to_three, flexiv_base),
        inverse(flexiv_to_three),
    )
    assert close(scene_base, (half, half, 0.0, 0.0))
    model_in_scene = multiply(scene_base, flexiv_to_three)
    assert close(model_in_scene, (0.5, 0.5, 0.5, 0.5))
    assert close(rotate(model_in_scene, (1, 0, 0)), (0, 1, 0))
    assert close(rotate(model_in_scene, (0, 1, 0)), (0, 0, 1))
    assert close(rotate(model_in_scene, (0, 0, 1)), (1, 0, 0))


def test_viewer_loads_distinct_rizon4_and_rizon4r_models_and_asset_maps():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert 'variant: "Rizon4"' in source
    assert 'variant: "Rizon4R"' in source
    assert 'flexiv_Rizon4_kinematics.urdf' in source
    assert 'flexiv_Rizon4R_kinematics.urdf' in source
    assert "this.armModels = { left: null, right: null }" in source
    assert "this.armAssets = { left: new Map(), right: new Map() }" in source
    assert "this.armModels[side] = parseURDF(urdfTexts[index])" in source
    assert "this.armAssets[side].set(relativePath" in source
    assert "const robotModel = this.armModels[side]" in source
    assert "const assets = this.armAssets[side]" in source
    assert "assets.get(visual.filename)" in source
    assert "new URL(relativePath, this.assetRoot).href" in source
    assert "meshes/Rizon4/visual/${basename}" not in source


def test_viewer_preserves_the_vendored_mesh_root_in_package_urls():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert "function _resolveMeshAssetKey(filename)" in source
    assert 'raw.replace(/^package:\\/\\//, "")' in source
    assert 'withoutScheme.startsWith("meshes/")' in source
    assert 'withoutScheme.indexOf("/meshes/")' in source
    assert 'return `meshes/${withoutScheme}`' in source
    assert "filename: normalized" in source
    assert "const normalized = _resolveMeshAssetKey(filename)" in source
    assert 'replace(/^package:\\/\\/[^/]+\\//, "")' not in source


def test_viewer_omits_arm_and_gripper_adjacent_text_overlays():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert '"左臂 / LEFT"' not in source
    assert '"右臂 / RIGHT"' not in source
    assert '"左 TCP / LEFT TCP"' not in source
    assert '"右 TCP / RIGHT TCP"' not in source
    assert "createArmIdentityMarker" not in source
    assert "createTcpIdentityLabel" not in source
    assert "identityLabels: Object.freeze({ left: null, right: null })" in source
    assert "tcpFrame.add(tcpAxes, tcpMarker);" in source
    assert "tcpMarker.userData.armSide = side;" in source
    assert "armBasePositions: Object.freeze" in source


def test_viewer_exposes_read_only_dual_arm_debug_snapshot():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert 'Object.defineProperty(window, "__armViewerDebug"' in source
    assert "installDebugGetter();" in source
    assert "get: () => activeViewer?.getDebugSnapshot() || EMPTY_DEBUG_SNAPSHOT" in source
    assert "activeViewer = viewer" in source
    assert "loadedArmCount:" in source
    assert "meshCounts: Object.freeze" in source
    assert "worldBounds: Object.freeze" in source
    assert "left: boundsSnapshot(leftRoot)" in source
    assert "right: boundsSnapshot(rightRoot)" in source
    assert "combined: boundsSnapshot(this.sceneRoot)" in source
    assert "tcpFrames: Object.freeze" in source
    assert "groundPlane: Object.freeze" in source
    assert "floorScene: this.floor ? vectorSnapshot(this.floor.position) : null" in source
    assert "gridScene: this.grid ? vectorSnapshot(this.grid.position) : null" in source
    assert "viewTarget: Object.freeze" in source
    assert "controlsScene: this.controls ? vectorSnapshot(this.controls.target) : null" in source
    assert "defaultScene: this._defaultView ? vectorSnapshot(this._defaultView.target) : null" in source
    assert "worldFramePosition:" in source


def test_viewer_maps_absolute_tcp_pose_to_local_axes_without_tool_offset():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert "TCP_AXIS_SIZE = 0.14" in source
    assert "new THREE.AxesHelper(TCP_AXIS_SIZE)" in source
    assert "this.sceneRoot.add(tcpFrame);" in source
    assert "absoluteTcpPoseToSceneTransform(pose, side)" in source
    assert "stationFromRobot.multiply(robotFromTcp)" in source
    assert "quaternion: flexivFrameRotationToThree(flexivRotation)" in source
    assert "C * R (not C * R * C^-1)" in source
    assert "frame.position.copy(sceneTransform.position);" in source
    assert "frame.quaternion.copy(sceneTransform.quaternion);" in source
    assert "arm.tcpFrame.visible = hasFreshAbsoluteTcp;" in source
    assert "this.updateTcpPoses(null, null);" in source
    assert "payload.tcp_poses" in source
    assert "Active TCP" in source
    assert "tcp_umi" not in source


def test_tcp_side_coloring_replaces_text_and_absolute_pose_does_not_inherit_arm_root():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert "new THREE.MeshStandardMaterial({ color, emissive: color" in source
    assert "tcpFrame.add(tcpAxes, tcpMarker);" in source
    assert "this.sceneRoot.add(tcpFrame);" in source
    assert "urdfFrame.add(tcpFrame);" not in source


def test_viewer_uses_config_tool_transform_only_as_flange_local_fk_frame():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert 'const configuredTools = payload.configured_tools' in source
    assert 'value.frame !== "flange"' in source
    assert 'quaternionOrder !== "wxyz"' in source
    assert 'configuredToolFrame.name = `${side}-configured-tool-frame`' in source
    assert 'const flangeLink = links.get("flange")' in source
    assert 'flangeLink.add(configuredToolFrame, configuredToolVisualFrame)' in source
    assert 'const [x, y, z] = transform.translation;' in source
    assert 'const [qw, qx, qy, qz] = transform.quaternion;' in source
    assert 'frame.quaternion.set(qx, qy, qz, qw).normalize();' in source
    assert 'CONFIG_TOOL_AXIS_SIZE = 0.12' in source
    assert 'new THREE.AxesHelper(CONFIG_TOOL_AXIS_SIZE)' in source
    assert 'flange→tool 坐标轴' in source


def test_both_urdfs_expose_a_seven_joint_chain_to_flange_for_tool_fk():
    for variant in ("Rizon4", "Rizon4R"):
        root = ET.parse(ASSET_ROOT / f"flexiv_{variant}_kinematics.urdf").getroot()
        joints_by_child = {
            joint.find("child").attrib["link"]: joint
            for joint in root.findall("joint")
            if joint.find("child") is not None
        }
        link = "flange"
        movable = []
        visited = set()
        while link in joints_by_child:
            assert link not in visited
            visited.add(link)
            joint = joints_by_child[link]
            if joint.attrib.get("type") != "fixed":
                movable.append(joint.attrib["name"])
            link = joint.find("parent").attrib["link"]
        assert list(reversed(movable)) == [f"joint{index}" for index in range(1, 8)]


def test_fresh_absolute_shm_tcp_overrides_config_frame_without_offset_composition():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert 'const hasFreshAbsoluteTcp = Boolean(this.tcpStates[side]);' in source
    assert 'arm.tcpFrame.visible = hasFreshAbsoluteTcp;' in source
    assert '!hasFreshAbsoluteTcp' in source
    assert 'arm.configuredToolFrame.userData.configured' in source
    assert 'this.updateConfiguredTools(' in source
    assert source.index('this.updateConfiguredTools(') < source.index(
        'this.updateTcpPoses(state.tcpPoses.left, state.tcpPoses.right)'
    )
    assert 'Active TCP（SHM absolute）' in source
    assert '${prefix} ${label}（实时 TCP 坐标轴覆盖）' in source
    assert '.multiply(transform' not in source
    assert '.compose(transform' not in source


def test_unknown_config_tool_pose_is_hidden_and_explained_not_fabricated():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert 'tool.status === "available" ? tool.transform : null' in source
    assert 'frame.visible = false;' in source
    assert 'controller-active-unknown' in source
    assert '控制器活动工具未知（config tool_name 为空）' in source
    assert 'name-known-pose-unknown' in source
    assert 'config 未提供 flange→tool 位姿' in source
    assert 'configuredToolFrames: Object.freeze' in source
    assert 'configuredTools: Object.freeze' in source


def test_elements_resolved_tool_and_asset_are_part_of_the_viewer_contract():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert "value.resolved_tool_name" in source
    assert "value.resolved_tool_source" in source
    assert 'resolution === "elements-active-tool"' in source
    assert 'path.startsWith("/static/")' in source
    assert 'kind === "urdf"' in source
    assert 'kind === "obj"' in source
    assert 'configuredToolVisualFrame.name = `${side}-configured-tool-visual-frame`' in source
    assert 'configuredToolPlaceholder = createConfiguredToolPlaceholder(side)' in source
    assert 'flangeLink.add(configuredToolFrame, configuredToolVisualFrame)' in source
    assert "arm.configuredToolVisualFrame.visible = visualVisible;" in source
    assert "function configuredToolVisualTransform(tool)" in source
    assert "if (tool?.visual?.mountTransform) return tool.visual.mountTransform;" in source
    assert 'Tool model failed (' in source
    assert 'placeholder shown' in source


def test_tool_asset_switch_is_generation_guarded_and_differential():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert "class ToolAssetRequestGate" in source
    assert "const epoch = this.toolAssetRequestGate.next(side);" in source
    assert "this._clearConfiguredToolAsset(side);" in source
    assert "this.toolAssetRequestGate.settle(" in source
    assert "if (this._disposed || !accepted)" in source
    assert "this._perf.tool_asset_stale_drops += 1" in source
    assert "this.toolAssetCache.has(key)" in source
    assert "frame.userData.toolSignature === signature" in source
    assert 'this._invalidateRender("configured-tool-asset-ready")' in source
    assert 'this._invalidateRender("configured-tool-asset-error")' in source
    assert "requestAnimationFrame" not in source[source.index("_updateConfiguredToolAsset"):source.index("_syncTcpFrameVisibility")]


def test_tool_urdf_loader_is_same_origin_and_mesh_failure_stays_local():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert "function toolUrdfMeshUrl(filename, urdfUrl)" in source
    assert 'globalThis.location?.href || fallbackBase.href' in source
    assert 'resolvedUrdfUrl.origin !== pageUrl.origin' in source
    assert 'url.origin !== pageUrl.origin || !url.pathname.startsWith("/static/")' in source
    assert 'void disposeToolAssetCache(this.toolAssetCache);' in source
    assert "Unsupported or unsafe tool URDF mesh" in source
    assert "buildDynamicUrdfObject(" in source
    assert 'lower.endsWith(".stl")' in source
    assert "function parseBinarySTL(input)" in source
    assert 'status: "error"' in source
    failure_block = source[source.index("if (error) {"):source.index("const mounted = cloneToolAssetSource")]
    assert "_setConfiguredToolPlaceholder(side, true)" in failure_block
    assert '_setState("error"' not in failure_block


def test_debug_snapshot_exposes_tool_asset_state_without_polling_more_often():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert "toolAssets: Object.freeze" in source
    assert "...this.toolAssetStates.left" in source
    assert "meshCount: meshCount(this.toolAssetObjects.left)" in source
    assert "tool_asset_loads" in source
    assert "tool_asset_failures" in source
    assert "tool_asset_stale_drops" in source
    assert "DEFAULT_STATE_POLL_INTERVAL_MS = 100" in source
    assert "MAX_RENDER_FPS = 30" in source


def test_joint_angles_use_quaternions_and_preserve_origin_orientation():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert ".rotation.setFromAxisAngle" not in source
    assert "motion.userData.baseQuaternion = motion.quaternion.clone();" in source
    assert "motion.userData.axisAngleQuaternion = new THREE.Quaternion();" in source
    assert "axisAngle.setFromAxisAngle(joint.userData.axis, value);" in source
    assert re.search(
        r"joint\.quaternion\.copy\(joint\.userData\.baseQuaternion\)\.multiply\(axisAngle\)",
        source,
    )
    assert source.index("transformFromURDF(joint.transform, jointGroup);") < source.index(
        "jointGroup.add(motion);"
    )


def test_viewer_polls_read_only_robot_state_api_and_handles_missing_data():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert 'DEFAULT_ROBOT_STATE_URL = "/api/robots/state"' in source
    assert "DEFAULT_STATE_POLL_INTERVAL_MS = 100" in source
    assert 'cache: "no-store"' in source
    assert "this.updateJointStates(state.left, state.right)" in source
    assert "home.joints_left" in source
    assert "home.joints_right" in source
    assert "left: liveLeft || homeLeft || [...DEFAULT_POSE]" in source
    assert "right: liveRight || homeRight || [...DEFAULT_POSE]" in source
    assert "this.homeJointStates.left || DEFAULT_POSE" in source
    assert "this.homeJointStates.right || DEFAULT_POSE" in source
    assert "this.updateJointStates(left, right)" in source
    assert "无实时数据" in source
    assert "console." not in source


def test_viewer_uses_dirty_capped_rendering_and_pauses_when_not_visible():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert "class DirtyRenderLoop" in source
    assert "MAX_RENDER_FPS = 30" in source
    assert "MAX_DEVICE_PIXEL_RATIO = 1.5" in source
    assert "this._renderLoop?.invalidate(reason)" in source
    assert "requestAnimationFrame(() => this._animate())" not in source
    assert 'document.addEventListener("visibilitychange"' in source
    assert "new IntersectionObserver" in source
    assert "this._renderLoop?.pause()" in source
    assert "this._scheduleStatePoll(0)" in source


def test_viewer_deduplicates_states_and_exposes_performance_snapshot():
    source = (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")
    assert "JOINT_UPDATE_EPSILON_RAD = 1e-4" in source
    assert "TCP_UPDATE_EPSILON = 1e-5" in source
    assert "numericVectorsNearlyEqual" in source
    assert "joint_skips" in source
    assert "tcp_skips" in source
    assert "tool_skips" in source
    assert "rendererInfoSnapshot(this.renderer)" in source
    assert "ghost_rebuilds" in source
    assert "READY_HOME_POSE_CANDIDATE_POLL_INTERVAL_MS = 5000" in source
