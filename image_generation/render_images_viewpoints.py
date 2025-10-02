"""Modernized CLEVR image generation with configurable camera viewpoints.

This script is inspired by the original ``render_images.py`` generator but it is
updated for Blender 3.x and introduces explicit control over camera
viewpoints.  Camera locations can either be sampled randomly from spherical
coordinates or loaded from a JSON file.  Each generated scene is saved alongside
its metadata in the same format as the classic CLEVR dataset.

Run the script from Blender like this::

    blender --background --python render_images_viewpoints.py -- [options]

The arguments that follow ``--`` are forwarded directly to this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from mathutils import Matrix, Vector

import utils


# -----------------------------------------------------------------------------
# Camera utilities


@dataclass(frozen=True)
class Viewpoint:
    """Simple representation of a camera viewpoint."""

    location: Vector
    look_at: Vector
    up: Vector

    @staticmethod
    def from_dict(data: Dict[str, Sequence[float]], default_up: Vector) -> "Viewpoint":
        try:
            location = Vector(data["location"])  # type: ignore[arg-type]
            look_at_raw = data.get("look_at", (0.0, 0.0, 0.0))
            up_raw = data.get("up", tuple(default_up))
            look_at = Vector(look_at_raw)
            up = Vector(up_raw)
        except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(
                "Camera viewpoints must define a 'location' (and optionally\n"
                "'look_at' and 'up') as sequences of 3 floats"
            ) from exc
        return Viewpoint(location=location, look_at=look_at, up=up)


def orient_camera(camera: bpy.types.Object, viewpoint: Viewpoint) -> None:
    """Place ``camera`` so that it looks at ``viewpoint.look_at``."""

    camera.location = viewpoint.location
    direction = (viewpoint.look_at - camera.location).normalized()

    # Create an orthonormal basis (right, up, forward) for the camera
    up = viewpoint.up.normalized()
    right = direction.cross(up)
    if right.length_squared == 0:  # Degenerate when up == direction
        # Fall back to a canonical up vector.
        up = Vector((0.0, 0.0, 1.0))
        right = direction.cross(up)
    right.normalize()
    corrected_up = right.cross(direction).normalized()

    # Blender camera looks along its -Z axis with Y as the up axis.
    rot = Matrix((
        (right.x, corrected_up.x, -direction.x, 0.0),
        (right.y, corrected_up.y, -direction.y, 0.0),
        (right.z, corrected_up.z, -direction.z, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))

    rot.translation = viewpoint.location
    camera.matrix_world = rot


def spherical_viewpoint(
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
    target: Vector,
    up: Vector,
) -> Viewpoint:
    """Generate a :class:`Viewpoint` from spherical coordinates."""

    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    x = radius * math.cos(azimuth_rad) * math.cos(elevation_rad)
    y = radius * math.sin(azimuth_rad) * math.cos(elevation_rad)
    z = radius * math.sin(elevation_rad)

    return Viewpoint(location=Vector((x, y, z)) + target, look_at=target, up=up)


# -----------------------------------------------------------------------------
# Rendering helpers


def _configure_cycles(args: argparse.Namespace, output_path: str) -> None:
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.filepath = output_path
    render.resolution_x = args.width
    render.resolution_y = args.height
    render.resolution_percentage = 100

    cycles = scene.cycles
    cycles.samples = args.render_num_samples
    cycles.use_adaptive_sampling = getattr(args, "use_adaptive_sampling", False)
    cycles.transparent_min_bounces = args.render_min_bounces
    cycles.transparent_max_bounces = args.render_max_bounces

    if hasattr(cycles, "tile_size"):
        cycles.tile_size = args.render_tile_size
    else:  # Blender 3.0+ separates tile dimensions
        if hasattr(render, "tile_x"):
            render.tile_x = args.render_tile_size
        if hasattr(render, "tile_y"):
            render.tile_y = args.render_tile_size

    if args.use_gpu:
        scene.cycles.device = "GPU"
        _enable_gpu(args)


def _enable_gpu(args: argparse.Namespace) -> None:
    """Enable GPU rendering for Cycles in Blender 2.9+/3.x."""

    prefs = bpy.context.preferences
    if "cycles" not in prefs.addons:
        bpy.ops.preferences.addon_enable(module="cycles")

    cycles_prefs = prefs.addons["cycles"].preferences
    backend = args.gpu_backend.upper()
    cycles_prefs.compute_device_type = backend

    # Force device discovery – required when running headless.
    try:
        cycles_prefs.get_devices()
    except AttributeError:
        # Older Blender versions use update() with a context argument.
        cycles_prefs.get_devices(bpy.context)

    # Enable every device that matches the backend; keep CPU disabled unless
    # explicitly requested.
    enabled = False
    for device in cycles_prefs.devices:
        use_device = device.type == backend or (
            backend == "OPTIX" and device.type in {"OPTIX", "CUDA"}
        )
        device.use = use_device
        enabled = enabled or use_device

    if not enabled:
        print(f"[WARN] No {backend} devices available; falling back to CPU rendering")
        bpy.context.scene.cycles.device = "CPU"


def _jitter_object(name: str, magnitude: float) -> None:
    if magnitude <= 0:
        return
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    for axis in range(3):
        obj.location[axis] += (random.random() - 0.5) * 2.0 * magnitude


def _compute_directions(camera: bpy.types.Object) -> Dict[str, Tuple[float, float, float]]:
    plane_normal = Vector((0.0, 0.0, 1.0))
    quat = camera.matrix_world.to_quaternion()
    cam_behind = quat @ Vector((0, 0, -1))
    cam_left = quat @ Vector((-1, 0, 0))
    cam_up = quat @ Vector((0, 1, 0))

    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    return {
        "behind": tuple(plane_behind),
        "front": tuple(-plane_behind),
        "left": tuple(plane_left),
        "right": tuple(-plane_left),
        "above": tuple(plane_up),
        "below": tuple(-plane_up),
    }


# -----------------------------------------------------------------------------
# Scene generation


def load_properties(args: argparse.Namespace) -> Tuple[
    Dict[str, List[int]],
    List[Tuple[str, float]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:
    with open(args.properties_json, "r") as handle:
        properties = json.load(handle)

    color_name_to_rgba: Dict[str, List[float]] = {}
    for name, rgb in properties["colors"].items():
        rgba = [float(c) / 255.0 for c in rgb] + [1.0]
        color_name_to_rgba[name] = rgba

    size_mapping = list(properties["sizes"].items())
    material_mapping = [(v, k) for k, v in properties["materials"].items()]
    object_mapping = [(v, k) for k, v in properties["shapes"].items()]

    return color_name_to_rgba, size_mapping, material_mapping, object_mapping


def add_random_objects(
    scene_struct: Dict,
    num_objects: int,
    args: argparse.Namespace,
    camera: bpy.types.Object,
) -> Tuple[List[Dict], List[bpy.types.Object]]:
    (
        color_name_to_rgba,
        size_mapping,
        material_mapping,
        object_mapping,
    ) = load_properties(args)

    shape_color_combos = None
    if args.shape_color_combos_json is not None:
        with open(args.shape_color_combos_json, "r") as handle:
            shape_color_combos = list(json.load(handle).items())

    positions: List[Tuple[float, float, float]] = []
    objects: List[Dict] = []
    blender_objects: List[bpy.types.Object] = []

    for _ in range(num_objects):
        size_name, r = random.choice(size_mapping)

        num_tries = 0
        while True:
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_random_objects(scene_struct, num_objects, args, camera)

            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)

            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.hypot(dx, dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                for direction_name in ["left", "right", "front", "behind"]:
                    direction_vec = scene_struct["directions"][direction_name]
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        if shape_color_combos is None:
            obj_name, obj_name_out = random.choice(object_mapping)
            color_name, rgba = random.choice(list(color_name_to_rgba.items()))
        else:
            obj_name_out, color_choices = random.choice(shape_color_combos)
            color_name = random.choice(color_choices)
            obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
            rgba = color_name_to_rgba[color_name]

        if obj_name == "Cube":
            r /= math.sqrt(2)

        theta = 360.0 * random.random()
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        mat_name, mat_name_out = random.choice(material_mapping)
        utils.add_material(mat_name, Color=rgba)

        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append(
            {
                "shape": obj_name_out,
                "size": size_name,
                "material": mat_name_out,
                "3d_coords": tuple(obj.location),
                "rotation": theta,
                "pixel_coords": pixel_coords,
                "color": color_name,
            }
        )

    return objects, blender_objects


def compute_all_relationships(scene_struct: Dict, eps: float = 0.2) -> Dict[str, List[List[int]]]:
    all_relationships = {}
    for name, direction_vec in scene_struct["directions"].items():
        if name in {"above", "below"}:
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct["objects"]):
            coords1 = obj1["3d_coords"]
            related: List[int] = []
            for j, obj2 in enumerate(scene_struct["objects"]):
                if i == j:
                    continue
                coords2 = obj2["3d_coords"]
                diff = [coords2[k] - coords1[k] for k in range(3)]
                dot = sum(diff[k] * direction_vec[k] for k in range(3))
                if dot > eps:
                    related.append(j)
            all_relationships[name].append(sorted(related))
    return all_relationships


def render_scene(
    args: argparse.Namespace,
    viewpoint: Viewpoint,
    num_objects: int,
    output_index: int,
    output_split: str,
    output_image: str,
    output_scene: str,
    output_blendfile: Optional[str] = None,
) -> None:
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    utils.load_materials(args.material_dir)

    camera = bpy.data.objects["Camera"]
    orient_camera(camera, viewpoint)

    _configure_cycles(args, output_image)

    # Lighting jitter
    _jitter_object("Lamp_Key", args.key_light_jitter)
    _jitter_object("Lamp_Back", args.back_light_jitter)
    _jitter_object("Lamp_Fill", args.fill_light_jitter)

    scene_struct = {
        "split": output_split,
        "image_index": output_index,
        "image_filename": os.path.basename(output_image),
        "objects": [],
        "directions": _compute_directions(camera),
    }

    objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera)
    scene_struct["objects"] = objects
    scene_struct["relationships"] = compute_all_relationships(scene_struct)

    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except RuntimeError:
            continue

    with open(output_scene, "w") as handle:
        json.dump(scene_struct, handle, indent=2)

    if output_blendfile is not None:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

    for obj in blender_objects:
        utils.delete_object(obj)


# -----------------------------------------------------------------------------
# Argument parsing & entry point


def parse_viewpoints(args: argparse.Namespace, count: int) -> List[Viewpoint]:
    default_up = Vector(args.camera_up)
    target = Vector(args.camera_target)

    if args.camera_viewpoints_json:
        with open(args.camera_viewpoints_json, "r") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list) or not payload:
            raise ValueError("camera_viewpoints_json must contain a non-empty list")
        viewpoints = [Viewpoint.from_dict(item, default_up) for item in payload]
        return viewpoints

    viewpoints: List[Viewpoint] = []
    for _ in range(count):
        radius = random.uniform(*args.camera_radius_range)
        azimuth = random.uniform(*args.camera_azimuth_range)
        elevation = random.uniform(*args.camera_elevation_range)
        viewpoints.append(spherical_viewpoint(radius, azimuth, elevation, target, default_up))
    return viewpoints


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--base_scene_blendfile", default="data/base_scene.blend")
    parser.add_argument("--properties_json", default="data/properties.json")
    parser.add_argument("--shape_dir", default="data/shapes")
    parser.add_argument("--material_dir", default="data/materials")
    parser.add_argument("--shape_color_combos_json", default=None)

    parser.add_argument("--min_objects", default=3, type=int)
    parser.add_argument("--max_objects", default=10, type=int)
    parser.add_argument("--min_dist", default=0.25, type=float)
    parser.add_argument("--margin", default=0.4, type=float)
    parser.add_argument("--min_pixels_per_object", default=200, type=int)
    parser.add_argument("--max_retries", default=50, type=int)

    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--num_images", default=5, type=int)
    parser.add_argument("--filename_prefix", default="CLEVR")
    parser.add_argument("--split", default="new")
    parser.add_argument("--output_image_dir", default="../output/images/")
    parser.add_argument("--output_scene_dir", default="../output/scenes/")
    parser.add_argument("--output_scene_file", default="../output/CLEVR_scenes.json")
    parser.add_argument("--output_blend_dir", default="output/blendfiles")
    parser.add_argument("--save_blendfiles", type=int, default=0)
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--license", default="Creative Commons Attribution (CC-BY 4.0)")
    parser.add_argument("--date", default="")

    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--gpu_backend", default="CUDA", choices=["CUDA", "OPTIX", "HIP", "METAL"])
    parser.add_argument("--width", default=320, type=int)
    parser.add_argument("--height", default=240, type=int)
    parser.add_argument("--key_light_jitter", default=1.0, type=float)
    parser.add_argument("--fill_light_jitter", default=1.0, type=float)
    parser.add_argument("--back_light_jitter", default=1.0, type=float)
    parser.add_argument("--render_num_samples", default=128, type=int)
    parser.add_argument("--render_min_bounces", default=8, type=int)
    parser.add_argument("--render_max_bounces", default=8, type=int)
    parser.add_argument("--render_tile_size", default=256, type=int)

    parser.add_argument("--camera_viewpoints_json", default=None)
    parser.add_argument("--camera_radius_range", nargs=2, type=float, default=[6.5, 8.5])
    parser.add_argument("--camera_azimuth_range", nargs=2, type=float, default=[35.0, 145.0])
    parser.add_argument("--camera_elevation_range", nargs=2, type=float, default=[20.0, 60.0])
    parser.add_argument("--camera_target", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--camera_up", nargs=3, type=float, default=[0.0, 0.0, 1.0])

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = utils.parse_args(parser, argv)

    if not args.date:
        from datetime import datetime

        args.date = datetime.today().strftime("%m/%d/%Y")

    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)
    if not os.path.isdir(args.output_scene_dir):
        os.makedirs(args.output_scene_dir)
    if args.save_blendfiles and not os.path.isdir(args.output_blend_dir):
        os.makedirs(args.output_blend_dir)

    num_digits = 6
    prefix = f"{args.filename_prefix}_{args.split}_"
    img_template = os.path.join(args.output_image_dir, f"{prefix}%0{num_digits}d.png")
    scene_template = os.path.join(args.output_scene_dir, f"{prefix}%0{num_digits}d.json")
    blend_template = os.path.join(args.output_blend_dir, f"{prefix}%0{num_digits}d.blend")

    viewpoints = parse_viewpoints(args, args.num_images)

    all_scene_paths: List[str] = []
    for i in range(args.num_images):
        img_path = img_template % (i + args.start_idx)
        scene_path = scene_template % (i + args.start_idx)
        blend_path = blend_template % (i + args.start_idx) if args.save_blendfiles else None

        viewpoint = viewpoints[i % len(viewpoints)] if args.camera_viewpoints_json else viewpoints[i]

        num_objects = random.randint(args.min_objects, args.max_objects)
        render_scene(
            args,
            viewpoint=viewpoint,
            num_objects=num_objects,
            output_index=i + args.start_idx,
            output_split=args.split,
            output_image=img_path,
            output_scene=scene_path,
            output_blendfile=blend_path,
        )
        all_scene_paths.append(scene_path)

    dataset = {
        "info": {
            "date": args.date,
            "version": args.version,
            "split": args.split,
            "license": args.license,
        },
        "scenes": [],
    }

    for path in all_scene_paths:
        with open(path, "r") as handle:
            dataset["scenes"].append(json.load(handle))

    with open(args.output_scene_file, "w") as handle:
        json.dump(dataset, handle)


if __name__ == "__main__":  # pragma: no cover - Blender entry point
    main()

