#!/usr/bin/env python3
"""Build the UMI3 wrist-camera hard/soft fingertip URDF variants.

Run this script with FreeCAD's Python interpreter. The delivered STEP file
names contain non-ASCII text, so sources are identified by their solid counts
instead of by file name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import FreeCAD as App
import Import
import MeshPart
import Part


MOUNT_X_MM = 125.814503
INDEX_PIVOT_CAD_MM = (-4.685497, 0.0, -21.21)
INDEX_ASSEMBLY_OFFSET_RAD = 0.11994605952189481
JOINT_UPPER_RAD = math.pi / 2.0
CONTACT_TOLERANCE_MM = 1e-5

# Body-relative component transforms extracted from the SolidWorks assembly.
# These matrices are expressed after Blender imported the OBJ geometry with
# Blender = diag(-1, -1, 1) * CAD.  Instance transforms must therefore be
# conjugated by that axis conversion before they are applied to STEP solids.
BLENDER_TO_CAD_AXIS = (-1.0, -1.0, 1.0)
CAMERA_BLENDER_TRANSFORM = (
    (0.0, -1.0, 0.0, -0.08627204748814458),
    (0.0, 0.0, -1.0, -0.004306474806139823),
    (1.0, 0.0, 0.0, -0.01254095684420557),
    (0.0, 0.0, 0.0, 1.0),
)
CAMERA_MOUNT_BLENDER_TRANSFORM = (
    (0.0, -1.0, 0.0, -0.06531450287586101),
    (0.0, 0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

VARIANTS = {
    "hard": {
        "finger_solid_count": 15,
        "base_features": {
            82,
            83,
            86,
            87,
            88,
            89,
            90,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
        },
        "finger_features": {0, 1, 2, 3, 4, 6, 7, 8},
        "fixed_contact_features": {87, 89, 90},
        "moving_contact_features": {3, 4, 6},
    },
    "soft": {
        "finger_solid_count": 17,
        "base_features": {
            82,
            83,
            86,
            87,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
        },
        "finger_features": {2, 3, 4, 6, 7, 14, 15, 16},
        "fixed_contact_features": {99, 100, 101},
        "moving_contact_features": {14, 15, 16},
    },
}


def feature_index(obj) -> int | None:
    match = re.fullmatch(r"Part__Feature(\d*)", obj.Name)
    if not match:
        return None
    return int(match.group(1) or "0")


def import_leaf_shapes(path: Path) -> Tuple[Dict[int, Part.Shape], Dict[int, str]]:
    doc = App.newDocument("umi3_mesh_source")
    try:
        Import.insert(str(path), doc.Name)
        doc.recompute()
        shapes: Dict[int, Part.Shape] = {}
        labels: Dict[int, str] = {}
        for obj in doc.Objects:
            index = feature_index(obj)
            if index is None or not hasattr(obj, "Shape") or obj.Shape.isNull():
                continue
            shapes[index] = obj.Shape.copy()
            labels[index] = obj.Label
        if not shapes:
            raise RuntimeError(f"STEP contains no leaf solids: {path}")
        return shapes, labels
    finally:
        App.closeDocument(doc.Name)


def discover_sources(directory: Path, finger_solid_count: int):
    candidates = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".step", ".stp"}
    )
    if not candidates:
        raise FileNotFoundError(f"No STEP files found under {directory}")

    by_count = {}
    for path in candidates:
        shapes, labels = import_leaf_shapes(path)
        count = len(shapes)
        if count in {1, 3, 102, finger_solid_count}:
            if count in by_count:
                raise RuntimeError(
                    f"Ambiguous {count}-solid STEP sources: {by_count[count][0]} and {path}"
                )
            by_count[count] = (path, shapes, labels)

    missing = {1, 3, 102, finger_solid_count} - set(by_count)
    if missing:
        raise RuntimeError(f"Missing STEP sources with solid counts: {sorted(missing)}")
    return {
        "body": by_count[102],
        "finger": by_count[finger_solid_count],
        "camera": by_count[3],
        "camera_mount": by_count[1],
    }


def require_features(shapes: Mapping[int, Part.Shape], indices: Iterable[int], label: str):
    missing = set(indices) - set(shapes)
    if missing:
        raise RuntimeError(f"{label} is missing feature indices {sorted(missing)}")


def compound(shapes: Mapping[int, Part.Shape], indices: Iterable[int]) -> Part.Shape:
    selected = [shapes[index] for index in sorted(indices)]
    if not selected:
        raise RuntimeError("Cannot create an empty compound")
    return Part.makeCompound(selected)


def cad_instance_matrix(rows: Sequence[Sequence[float]]) -> App.Matrix:
    matrix = App.Matrix()
    for row in range(3):
        for column in range(3):
            setattr(matrix, f"A{row + 1}{column + 1}", float(rows[row][column]))
        setattr(matrix, f"A{row + 1}4", float(rows[row][3]) * 1000.0)
    matrix.A41, matrix.A42, matrix.A43, matrix.A44 = 0.0, 0.0, 0.0, 1.0
    return matrix


def blender_instance_to_cad(
    rows: Sequence[Sequence[float]],
) -> Tuple[Tuple[float, float, float, float], ...]:
    """Convert a Blender-OBJ instance matrix back to native CAD axes."""

    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("Expected a 4x4 homogeneous instance matrix")
    axes = BLENDER_TO_CAD_AXIS
    converted = []
    for row in range(3):
        converted.append(
            tuple(
                axes[row] * float(rows[row][column]) * axes[column]
                for column in range(3)
            )
            + (axes[row] * float(rows[row][3]),)
        )
    converted.append((0.0, 0.0, 0.0, 1.0))
    return tuple(converted)


def transformed(shape: Part.Shape, matrix: App.Matrix) -> Part.Shape:
    result = shape.copy()
    result.transformShape(matrix, True, False)
    return result


def shape_bounds(shape: Part.Shape) -> dict:
    box = shape.BoundBox
    return {
        "min": [box.XMin, box.YMin, box.ZMin],
        "max": [box.XMax, box.YMax, box.ZMax],
    }


def cad_to_urdf_matrix(link_origin_cad_mm: Sequence[float]) -> App.Matrix:
    ox, oy, oz = link_origin_cad_mm
    matrix = App.Matrix()
    # URDF x = CAD z - origin z
    matrix.A11, matrix.A12, matrix.A13, matrix.A14 = 0.0, 0.0, 1.0, -oz
    # URDF y = CAD y - origin y
    matrix.A21, matrix.A22, matrix.A23, matrix.A24 = 0.0, 1.0, 0.0, -oy
    # URDF z = origin x - CAD x
    matrix.A31, matrix.A32, matrix.A33, matrix.A34 = -1.0, 0.0, 0.0, ox
    matrix.A41, matrix.A42, matrix.A43, matrix.A44 = 0.0, 0.0, 0.0, 1.0
    return matrix


def rotate_about(shape: Part.Shape, angle_rad: float) -> Part.Shape:
    pivot = App.Vector(*INDEX_PIVOT_CAD_MM)
    matrix = App.Rotation(App.Vector(0, 1, 0), math.degrees(angle_rad)).toMatrix()
    rotated_pivot = matrix.multVec(pivot)
    matrix.A14 = pivot.x - rotated_pivot.x
    matrix.A24 = pivot.y - rotated_pivot.y
    matrix.A34 = pivot.z - rotated_pivot.z
    return transformed(shape, matrix)


def distance_at(fixed: Part.Shape, moving: Part.Shape, angle_rad: float) -> float:
    return float(fixed.distToShape(rotate_about(moving, angle_rad))[0])


def first_contact_angle(fixed: Part.Shape, moving: Part.Shape) -> float:
    lower = INDEX_ASSEMBLY_OFFSET_RAD
    upper = INDEX_ASSEMBLY_OFFSET_RAD + 1.5
    if distance_at(fixed, moving, lower) <= CONTACT_TOLERANCE_MM:
        return lower
    if distance_at(fixed, moving, upper) > CONTACT_TOLERANCE_MM:
        raise RuntimeError("Fingertips do not reach contact within the expected sweep")
    for _ in range(42):
        middle = (lower + upper) / 2.0
        if distance_at(fixed, moving, middle) > CONTACT_TOLERANCE_MM:
            lower = middle
        else:
            upper = middle
    return upper


def export_mesh(
    shape: Part.Shape,
    path: Path,
    *,
    linear_deflection: float,
    angular_deflection: float,
) -> dict:
    mesh = MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=linear_deflection,
        AngularDeflection=angular_deflection,
        Relative=False,
    )
    mesh.removeDuplicatedPoints()
    if mesh.CountFacets <= 0:
        raise RuntimeError(f"Tessellation produced no facets for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(str(path))
    box = mesh.BoundBox
    result = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "facets": mesh.CountFacets,
        "points": mesh.CountPoints,
        "bbox_mm": {
            "min": [box.XMin, box.YMin, box.ZMin],
            "max": [box.XMax, box.YMax, box.ZMax],
        },
    }
    mesh.clear()
    return result


def urdf_text(variant: str, joint_origin_rad: float) -> str:
    return f'''<?xml version="1.0"?>
<robot name="umi3_camera_{variant}_finger">
  <material name="body_dark"><color rgba="0.16 0.18 0.21 1.0"/></material>
  <material name="finger_{variant}"><color rgba="0.03 0.52 0.58 1.0"/></material>
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.09" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <geometry><mesh filename="meshes/base_link.stl" scale="0.001 0.001 0.001"/></geometry>
      <material name="body_dark"/>
    </visual>
  </link>
  <link name="index_finger_link">
    <inertial>
      <origin xyz="-0.03 0 0.02" rpy="0 0 0"/>
      <mass value="0.12"/>
      <inertia ixx="0.00006" ixy="0" ixz="0" iyy="0.00010" iyz="0" izz="0.00008"/>
    </inertial>
    <visual>
      <geometry><mesh filename="meshes/index_link.stl" scale="0.001 0.001 0.001"/></geometry>
      <material name="finger_{variant}"/>
    </visual>
  </link>
  <joint name="gripper_joint" type="revolute">
    <origin xyz="-0.021210000 0 0.130500000" rpy="0 {joint_origin_rad:.12f} 0"/>
    <parent link="base_link"/>
    <child link="index_finger_link"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0.0" upper="{JOINT_UPPER_RAD:.12f}" effort="100.0" velocity="1.5"/>
    <dynamics damping="0.05" friction="0.02"/>
  </joint>
</robot>
'''


def build_variant(variant: str, directory: Path, output_root: Path) -> dict:
    spec = VARIANTS[variant]
    sources = discover_sources(directory, spec["finger_solid_count"])
    _, body_shapes, body_labels = sources["body"]
    _, finger_shapes, finger_labels = sources["finger"]
    camera_path, camera_shapes, _ = sources["camera"]
    mount_path, mount_shapes, _ = sources["camera_mount"]

    for indices, shapes, label in (
        (spec["base_features"], body_shapes, f"{variant} base"),
        (spec["finger_features"], finger_shapes, f"{variant} finger"),
        (spec["fixed_contact_features"], body_shapes, f"{variant} fixed contact"),
        (spec["moving_contact_features"], finger_shapes, f"{variant} moving contact"),
    ):
        require_features(shapes, indices, label)

    fixed_contact = compound(body_shapes, spec["fixed_contact_features"])
    moving_contact = compound(finger_shapes, spec["moving_contact_features"])
    joint_origin_rad = first_contact_angle(fixed_contact, moving_contact)
    closed_distance = distance_at(fixed_contact, moving_contact, joint_origin_rad)
    open_distance = distance_at(
        fixed_contact,
        moving_contact,
        joint_origin_rad - JOINT_UPPER_RAD,
    )
    if closed_distance > CONTACT_TOLERANCE_MM:
        raise RuntimeError(f"{variant} q=0 contact distance is {closed_distance} mm")
    if open_distance <= 1.0:
        raise RuntimeError(f"{variant} q=pi/2 opening is only {open_distance} mm")

    body_shape = compound(body_shapes, spec["base_features"])
    camera_cad_transform = blender_instance_to_cad(CAMERA_BLENDER_TRANSFORM)
    mount_cad_transform = blender_instance_to_cad(
        CAMERA_MOUNT_BLENDER_TRANSFORM
    )
    camera_matrix = cad_instance_matrix(camera_cad_transform)
    mount_matrix = cad_instance_matrix(mount_cad_transform)
    camera_shape = transformed(
        Part.makeCompound(list(camera_shapes.values())), camera_matrix
    )
    mount_shape = transformed(
        Part.makeCompound(list(mount_shapes.values())), mount_matrix
    )
    base_to_urdf = cad_to_urdf_matrix((MOUNT_X_MM, 0.0, 0.0))
    component_shapes_urdf = {
        "body": transformed(body_shape, base_to_urdf),
        "camera_mount": transformed(mount_shape, base_to_urdf),
        "camera": transformed(camera_shape, base_to_urdf),
    }
    base_shape = Part.makeCompound(list(component_shapes_urdf.values()))
    finger_shape = transformed(
        compound(finger_shapes, spec["finger_features"]),
        cad_to_urdf_matrix(INDEX_PIVOT_CAD_MM),
    )

    variant_root = output_root / variant
    mesh_root = variant_root / "meshes"
    mesh_records = {
        "base_link.stl": export_mesh(
            base_shape,
            mesh_root / "base_link.stl",
            linear_deflection=0.28,
            angular_deflection=0.32,
        ),
        "index_link.stl": export_mesh(
            finger_shape,
            mesh_root / "index_link.stl",
            linear_deflection=0.22,
            angular_deflection=0.28,
        ),
    }
    for name, record in mesh_records.items():
        record["path"] = f"{variant}/meshes/{name}"
    urdf_path = variant_root / "umi3_gripper.urdf"
    urdf_path.write_text(urdf_text(variant, joint_origin_rad), encoding="utf-8", newline="\n")

    return {
        "variant": variant,
        "sources": {
            key: {
                "file": value[0].name,
                "sha256": hashlib.sha256(value[0].read_bytes()).hexdigest(),
                "solid_count": len(value[1]),
            }
            for key, value in sources.items()
        },
        "base_features": sorted(spec["base_features"]),
        "finger_features": sorted(spec["finger_features"]),
        "selected_labels": {
            "base": [body_labels[index] for index in sorted(spec["base_features"])],
            "finger": [finger_labels[index] for index in sorted(spec["finger_features"])],
        },
        "joint": {
            "name": "gripper_joint",
            "lower_rad": 0.0,
            "upper_rad": JOINT_UPPER_RAD,
            "closed_origin_rad": joint_origin_rad,
            "closed_contact_distance_mm": closed_distance,
            "open_contact_distance_mm": open_distance,
        },
        "urdf": {
            "path": f"{variant}/umi3_gripper.urdf",
            "bytes": urdf_path.stat().st_size,
            "sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        },
        "meshes": mesh_records,
        "component_transforms": {
            "camera": {
                "source_blender": CAMERA_BLENDER_TRANSFORM,
                "cad": camera_cad_transform,
            },
            "camera_mount": {
                "source_blender": CAMERA_MOUNT_BLENDER_TRANSFORM,
                "cad": mount_cad_transform,
            },
        },
        "component_bounds_urdf_mm": {
            name: shape_bounds(shape)
            for name, shape in component_shapes_urdf.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-dir", type=Path, required=True)
    parser.add_argument("--soft-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "static"
        / "assets"
        / "tools"
        / "umi3_camera",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    records = {
        variant: build_variant(
            variant,
            args.hard_dir if variant == "hard" else args.soft_dir,
            args.output_root,
        )
        for variant in ("hard", "soft")
    }
    for source_key in ("camera", "camera_mount"):
        if (
            records["hard"]["sources"][source_key]["sha256"]
            != records["soft"]["sources"][source_key]["sha256"]
        ):
            raise RuntimeError(f"Hard/soft {source_key} STEP sources differ")
    manifest = {
        "schema": "superinference.umi3-camera-visual.v2",
        "coordinate_conversion": {
            "urdf_x": "cad_z",
            "urdf_y": "cad_y",
            "urdf_z": "mount_x_minus_cad_x",
            "mount_x_mm": MOUNT_X_MM,
            "blender_obj_from_cad_axis": BLENDER_TO_CAD_AXIS,
            "instance_conversion": "M_cad = D * M_blender * D",
        },
        "variants": records,
    }
    manifest_path = args.output_root / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
