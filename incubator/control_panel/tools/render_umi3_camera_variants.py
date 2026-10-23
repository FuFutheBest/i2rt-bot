#!/usr/bin/env python3
"""Render closed/open QA images for both generated UMI3 variants in Blender."""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from mathutils import Vector


ASSET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "static"
    / "assets"
    / "tools"
    / "umi3_camera"
)


def cli_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "_qa"
        / "umi3_camera_previews",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            collection.remove(item)


def material(name, color, *, metallic=0.0, roughness=0.45):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1.0)
    value.use_nodes = True
    shader = next(node for node in value.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return value


def import_stl(path: Path, name: str, assigned_material):
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.stl_import(filepath=str(path))
    imported = list(set(bpy.context.scene.objects) - before)
    if len(imported) != 1:
        raise RuntimeError(f"Expected one object from {path}, got {len(imported)}")
    obj = imported[0]
    obj.name = name
    obj.data.materials.append(assigned_material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def joint_origin(urdf_path: Path) -> float:
    joint = ET.parse(urdf_path).getroot().find("joint[@name='gripper_joint']")
    if joint is None:
        raise RuntimeError(f"Missing gripper_joint in {urdf_path}")
    return float(joint.find("origin").attrib["rpy"].split()[1])


def build_scene(asset_root: Path, variant: str, joint_angle: float):
    clear_scene()
    variant_root = asset_root / variant
    body_material = material("Body", (0.10, 0.12, 0.15), metallic=0.55, roughness=0.28)
    finger_color = (0.05, 0.64, 0.67) if variant == "soft" else (0.22, 0.55, 0.86)
    finger_material = material("Finger", finger_color, metallic=0.18, roughness=0.36)
    base = import_stl(variant_root / "meshes" / "base_link.stl", "base_link", body_material)
    finger = import_stl(
        variant_root / "meshes" / "index_link.stl",
        "index_finger_link",
        finger_material,
    )
    finger.location = (-21.21, 0.0, 130.5)
    finger.rotation_euler[1] = joint_origin(variant_root / "umi3_gripper.urdf") - joint_angle

    center = Vector((35.0, -35.0, 128.0))
    bpy.ops.object.camera_add(location=(330.0, -430.0, 275.0))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 340.0
    look_at(camera, center)
    bpy.context.scene.camera = camera

    for location, energy, size in (
        ((220.0, -260.0, 420.0), 1450.0, 250.0),
        ((-260.0, -100.0, 180.0), 900.0, 210.0),
        ((120.0, 240.0, 240.0), 1050.0, 180.0),
    ):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        look_at(light, center)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 700
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.012, 0.014, 0.018)
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.7
    return scene, base, finger


def main():
    args = cli_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("soft", "hard"):
        for pose, angle in (("closed", 0.0), ("open", math.pi / 2.0)):
            scene, _, _ = build_scene(args.asset_root, variant, angle)
            output = args.output_dir / f"{variant}_{pose}.png"
            scene.render.filepath = str(output)
            bpy.ops.render.render(write_still=True)
            print(f"Rendered {output}")


if __name__ == "__main__":
    main()
