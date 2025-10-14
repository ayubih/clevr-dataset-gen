"""Modern CLEVR image generator with explicit camera viewpoint control.

This script is inspired by the original ``render_images.py`` but refreshes the
workflow for Blender 3.x/4.x.  Users can optionally provide a JSON file with a
list of camera viewpoints which will be applied sequentially (or randomly) when
rendering.  Cycles is configured with modern GPU settings so large batches of
images can be produced quickly.

Run from Blender:

```
blender --background --python render_images_modern.py -- [args]
```
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

INSIDE_BLENDER = True
try:  # pragma: no cover - Blender modules are only available inside Blender
    import bpy
    import bpy_extras
    from mathutils import Vector
except ImportError:  # pragma: no cover - fallback for help/usage outside Blender
    INSIDE_BLENDER = False
    bpy = None  # type: ignore
    bpy_extras = None  # type: ignore
    Vector = None  # type: ignore

if INSIDE_BLENDER:
    import utils


@dataclass(frozen=True)
class CameraViewpoint:
    """Description of a camera pose."""

    location: Tuple[float, float, float]
    look_at: Tuple[float, float, float]
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)


parser = argparse.ArgumentParser(description=__doc__)

# Input options
parser.add_argument(
    "--base_scene_blendfile",
    default="data/base_scene.blend",
    help=(
        "Base blender file on which all scenes are based; includes ground plane,"
        " lights, and camera."
    ),
)
parser.add_argument(
    "--properties_json",
    default="data/properties.json",
    help="JSON file defining objects, materials, sizes, and colors.",
)
parser.add_argument(
    "--shape_dir",
    default="data/shapes",
    help="Directory where .blend files for object models are stored",
)
parser.add_argument(
    "--material_dir",
    default="data/materials",
    help="Directory where .blend files for materials are stored",
)
parser.add_argument(
    "--shape_color_combos_json",
    default=None,
    help=(
        "Optional JSON mapping shape names to allowed colors (CLEVR-CoGenT"
        " style rendering)."
    ),
)

# Scene content
parser.add_argument(
    "--min_objects",
    default=3,
    type=int,
    help="Minimum number of objects to place in each scene",
)
parser.add_argument(
    "--max_objects",
    default=10,
    type=int,
    help="Maximum number of objects to place in each scene",
)
parser.add_argument(
    "--min_dist",
    default=0.25,
    type=float,
    help="Minimum allowed distance between object centers",
)
parser.add_argument(
    "--margin",
    default=0.4,
    type=float,
    help="Margin enforced along cardinal directions between all objects",
)
parser.add_argument(
    "--min_pixels_per_object",
    default=200,
    type=int,
    help="Minimum visible pixels per object; objects below this trigger a retry",
)
parser.add_argument(
    "--max_retries",
    default=50,
    type=int,
    help="Number of placement retries before the scene is regenerated",
)

# Output configuration
parser.add_argument(
    "--start_idx",
    default=0,
    type=int,
    help="Index offset used when naming rendered images",
)
parser.add_argument(
    "--num_images",
    default=5,
    type=int,
    help="Number of images to render",
)
parser.add_argument(
    "--filename_prefix",
    default="CLEVR",
    help="Prefix used when naming rendered images and JSON scene files",
)
parser.add_argument(
    "--split",
    default="new",
    help="Name of the dataset split to embed in the generated metadata",
)
parser.add_argument(
    "--output_image_dir",
    default="../output/images/",
    help="Directory where rendered images are written",
)
parser.add_argument(
    "--output_scene_dir",
    default="../output/scenes/",
    help="Directory where scene JSON files are written",
)
parser.add_argument(
    "--output_scene_file",
    default="../output/CLEVR_scenes.json",
    help="Location of the combined scene metadata file",
)
parser.add_argument(
    "--output_blend_dir",
    default="output/blendfiles",
    help="Directory used when saving intermediate .blend files",
)
parser.add_argument(
    "--save_blendfiles",
    type=int,
    default=0,
    help="Save a .blend snapshot for each render when set to 1",
)
parser.add_argument(
    "--version",
    default="2.0",
    help="Version string stored alongside the generated metadata",
)
parser.add_argument(
    "--license",
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="License stored in the metadata",
)
parser.add_argument(
    "--date",
    default=dt.today().strftime("%m/%d/%Y"),
    help="Date stored in the metadata",
)

# Rendering options
parser.add_argument(
    "--use_gpu",
    action="store_true",
    help="Enable GPU-accelerated Cycles rendering",
)
parser.add_argument(
    "--gpu_device_type",
    default="CUDA",
    choices=["CUDA", "OPTIX", "HIP", "METAL", "NONE"],
    help="Cycles compute device type to activate when --use_gpu is set",
)
parser.add_argument(
    "--gpu_device_name",
    default=None,
    help="Optional substring used to select a specific GPU device",
)
parser.add_argument(
    "--width",
    default=320,
    type=int,
    help="Output image width in pixels",
)
parser.add_argument(
    "--height",
    default=240,
    type=int,
    help="Output image height in pixels",
)
parser.add_argument(
    "--render_num_samples",
    default=128,
    type=int,
    help="Number of Cycles samples used during rendering",
)
parser.add_argument(
    "--render_min_bounces",
    default=4,
    type=int,
    help="Minimum number of light bounces",
)
parser.add_argument(
    "--render_max_bounces",
    default=8,
    type=int,
    help="Maximum number of light bounces",
)
parser.add_argument(
    "--render_tile_size",
    default=512,
    type=int,
    help="Tile size used by Cycles; larger tiles are typically faster on GPUs",
)
parser.add_argument(
    "--enable_adaptive_sampling",
    action="store_true",
    help="Enable adaptive sampling in Cycles (recommended for GPU rendering)",
)
parser.add_argument(
    "--cycles_denoiser",
    default=None,
    choices=[None, "OPTIX", "OPENIMAGEDENOISE", "NLM"],
    help="Optional Cycles denoiser to enable",
)

# Camera configuration
parser.add_argument(
    "--camera_jitter",
    default=0.0,
    type=float,
    help="Random jitter applied to the camera location (legacy behaviour)",
)
parser.add_argument(
    "--camera_focus_point",
    nargs=3,
    type=float,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="World-space point the camera looks at when no explicit look-at is provided",
)
parser.add_argument(
    "--camera_viewpoints_json",
    default=None,
    help="Path to a JSON file describing an ordered list of camera viewpoints",
)
parser.add_argument(
    "--camera_viewpoint_strategy",
    default="sequential",
    choices=["sequential", "random"],
    help="How to iterate through supplied camera viewpoints",
)
parser.add_argument(
    "--key_light_jitter",
    default=0.5,
    type=float,
    help="Random jitter applied to the key light position",
)
parser.add_argument(
    "--fill_light_jitter",
    default=0.5,
    type=float,
    help="Random jitter applied to the fill light position",
)
parser.add_argument(
    "--back_light_jitter",
    default=0.5,
    type=float,
    help="Random jitter applied to the back light position",
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _rand_range(scale: float) -> float:
    return 2.0 * scale * (random.random() - 0.5)


def _configure_cycles(args: argparse.Namespace) -> None:
    """Configure Cycles for Blender 3.x/4.x with optional GPU acceleration."""

    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = args.width
    render.resolution_y = args.height
    render.resolution_percentage = 100
    render.use_persistent_data = args.use_gpu

    # Cycles settings
    cycles_settings = scene.cycles
    cycles_settings.samples = args.render_num_samples
    cycles_settings.min_bounces = args.render_min_bounces
    cycles_settings.max_bounces = args.render_max_bounces
    cycles_settings.use_adaptive_sampling = args.enable_adaptive_sampling
    if hasattr(cycles_settings, "tile_x"):
        cycles_settings.tile_x = args.render_tile_size
        cycles_settings.tile_y = args.render_tile_size
    cycles_settings.blur_glossy = 2.0

    world = scene.world
    if world and hasattr(world, "cycles"):
        world.cycles.sample_as_light = True

    if args.cycles_denoiser:
        cycles_settings.use_denoising = True
        cycles_settings.denoiser = args.cycles_denoiser
    else:
        cycles_settings.use_denoising = False

    if args.use_gpu:
        preferences = bpy.context.preferences
        cycles_prefs = preferences.addons["cycles"].preferences
        if hasattr(cycles_prefs, "refresh_devices"):
            cycles_prefs.refresh_devices()
        cycles_prefs.compute_device_type = args.gpu_device_type
        devices = cycles_prefs.get_devices()
        target_name = args.gpu_device_name.lower() if args.gpu_device_name else None
        for device in devices:
            use_device = device.type in {"CUDA", "OPTIX", "HIP", "METAL", "GPU"}
            if target_name:
                use_device = use_device and target_name in device.name.lower()
            device.use = use_device
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"


def _load_camera_viewpoints(args: argparse.Namespace) -> Optional[List[CameraViewpoint]]:
    if not args.camera_viewpoints_json:
        return None
    path = Path(args.camera_viewpoints_json)
    with path.open("r", encoding="utf8") as handle:
        payload = json.load(handle)
    viewpoints: List[CameraViewpoint] = []
    for index, entry in enumerate(payload):
        if "location" not in entry:
            raise ValueError(f"Camera viewpoint {index} is missing a location")
        location = tuple(float(v) for v in entry["location"])
        look_at = tuple(float(v) for v in entry.get("look_at", args.camera_focus_point))
        up = tuple(float(v) for v in entry.get("up", (0.0, 0.0, 1.0)))
        viewpoints.append(CameraViewpoint(location, look_at, up))
    if not viewpoints:
        raise ValueError("Camera viewpoint file contained no entries")
    return viewpoints


def _select_viewpoint(
    viewpoints: Optional[List[CameraViewpoint]],
    args: argparse.Namespace,
    image_index: int,
) -> Optional[CameraViewpoint]:
    if not viewpoints:
        return None
    if args.camera_viewpoint_strategy == "random":
        return random.choice(viewpoints)
    idx = image_index % len(viewpoints)
    return viewpoints[idx]


def _point_camera(
    camera: bpy.types.Object,
    viewpoint: CameraViewpoint,
    jitter: float,
) -> None:
    camera.location = Vector(viewpoint.location)
    if jitter:
        camera.location += Vector((_rand_range(jitter), _rand_range(jitter), _rand_range(jitter)))
    focus = Vector(viewpoint.look_at)
    direction = focus - camera.location
    if direction.length == 0:
        direction = Vector((0.0, 0.0, -1.0))
    rotation = direction.to_track_quat("-Z", "Y").to_euler()
    camera.rotation_euler = rotation


# ---------------------------------------------------------------------------
# Core scene generation logic
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    if not INSIDE_BLENDER:
        raise RuntimeError("This script must be executed from within Blender")

    viewpoints = _load_camera_viewpoints(args)

    num_digits = 6
    prefix = f"{args.filename_prefix}_{args.split}_"
    img_template = f"{prefix}%0{num_digits}d.png"
    scene_template = f"{prefix}%0{num_digits}d.json"
    blend_template = f"{prefix}%0{num_digits}d.blend"

    image_dir = Path(args.output_image_dir)
    scene_dir = Path(args.output_scene_dir)
    blend_dir = Path(args.output_blend_dir)
    _ensure_directory(image_dir)
    _ensure_directory(scene_dir)
    if args.save_blendfiles:
        _ensure_directory(blend_dir)

    scene_paths: List[Path] = []
    for i in range(args.num_images):
        img_path = image_dir / (img_template % (i + args.start_idx))
        scene_path = scene_dir / (scene_template % (i + args.start_idx))
        scene_paths.append(scene_path)
        blend_path = None
        if args.save_blendfiles:
            blend_path = blend_dir / (blend_template % (i + args.start_idx))

        num_objects = random.randint(args.min_objects, args.max_objects)
        viewpoint = _select_viewpoint(viewpoints, args, i + args.start_idx)
        render_scene(
            args,
            num_objects=num_objects,
            output_index=i + args.start_idx,
            output_split=args.split,
            output_image=str(img_path),
            output_scene=str(scene_path),
            output_blendfile=str(blend_path) if blend_path else None,
            viewpoint=viewpoint,
        )

    all_scenes = []
    for scene_path in scene_paths:
        with scene_path.open("r", encoding="utf8") as handle:
            all_scenes.append(json.load(handle))

    combined = {
        "info": {
            "date": args.date,
            "version": args.version,
            "split": args.split,
            "license": args.license,
        },
        "scenes": all_scenes,
    }
    with Path(args.output_scene_file).open("w", encoding="utf8") as handle:
        json.dump(combined, handle)


def render_scene(
    args: argparse.Namespace,
    num_objects: int,
    output_index: int,
    output_split: str,
    output_image: str,
    output_scene: str,
    output_blendfile: Optional[str],
    viewpoint: Optional[CameraViewpoint],
) -> None:
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    utils.load_materials(args.material_dir)

    _configure_cycles(args)
    bpy.context.scene.render.filepath = output_image

    camera = bpy.data.objects["Camera"]
    if viewpoint:
        _point_camera(camera, viewpoint, args.camera_jitter)
    elif args.camera_jitter:
        for axis in range(3):
            camera.location[axis] += _rand_range(args.camera_jitter)

    scene_struct = {
        "split": output_split,
        "image_index": output_index,
        "image_filename": os.path.basename(output_image),
        "objects": [],
        "directions": {},
    }

    bpy.ops.mesh.primitive_plane_add(size=10)
    plane = bpy.context.object

    quat = camera.matrix_world.to_quaternion()
    cam_behind = quat @ Vector((0, 0, -1))
    cam_left = quat @ Vector((-1, 0, 0))
    cam_up = quat @ Vector((0, 1, 0))
    plane_normal = plane.data.vertices[0].normal
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()
    utils.delete_object(plane)

    scene_struct["directions"]["behind"] = tuple(plane_behind)
    scene_struct["directions"]["front"] = tuple(-plane_behind)
    scene_struct["directions"]["left"] = tuple(plane_left)
    scene_struct["directions"]["right"] = tuple(-plane_left)
    scene_struct["directions"]["above"] = tuple(plane_up)
    scene_struct["directions"]["below"] = tuple(-plane_up)

    if args.key_light_jitter:
        lamp = bpy.data.objects.get("Lamp_Key")
        if lamp:
            for axis in range(3):
                lamp.location[axis] += _rand_range(args.key_light_jitter)
    if args.back_light_jitter:
        lamp = bpy.data.objects.get("Lamp_Back")
        if lamp:
            for axis in range(3):
                lamp.location[axis] += _rand_range(args.back_light_jitter)
    if args.fill_light_jitter:
        lamp = bpy.data.objects.get("Lamp_Fill")
        if lamp:
            for axis in range(3):
                lamp.location[axis] += _rand_range(args.fill_light_jitter)

    objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera)

    scene_struct["objects"] = objects
    scene_struct["relationships"] = compute_all_relationships(scene_struct)

    bpy.ops.render.render(write_still=True)

    with open(output_scene, "w", encoding="utf8") as handle:
        json.dump(scene_struct, handle, indent=2)

    if output_blendfile:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


# ---------------------------------------------------------------------------
# Object placement utilities (adapted from the original CLEVR code)
# ---------------------------------------------------------------------------


def add_random_objects(
    scene_struct: dict,
    num_objects: int,
    args: argparse.Namespace,
    camera: bpy.types.Object,
) -> Tuple[List[dict], List[bpy.types.Object]]:
    with open(args.properties_json, "r", encoding="utf8") as handle:
        properties = json.load(handle)
        color_name_to_rgba = {
            name: [float(c) / 255.0 for c in rgb] + [1.0]
            for name, rgb in properties["colors"].items()
        }
        material_mapping = [(v, k) for k, v in properties["materials"].items()]
        object_mapping = [(v, k) for k, v in properties["shapes"].items()]
        size_mapping = list(properties["sizes"].items())

    shape_color_combos: Optional[List[Tuple[str, Sequence[str]]]] = None
    if args.shape_color_combos_json:
        with open(args.shape_color_combos_json, "r", encoding="utf8") as handle:
            shape_color_combos = list(json.load(handle).items())

    positions: List[Tuple[float, float, float]] = []
    objects: List[dict] = []
    blender_objects: List[bpy.types.Object] = []

    for _ in range(num_objects):
        size_name, radius = random.choice(size_mapping)

        num_tries = 0
        while True:
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_random_objects(scene_struct, num_objects, args, camera)

            x = random.uniform(-3.0, 3.0)
            y = random.uniform(-3.0, 3.0)
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - radius - rr < args.min_dist:
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
            radius /= math.sqrt(2)

        theta = 360.0 * random.random()
        utils.add_object(args.shape_dir, obj_name, radius, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, radius))

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


def compute_all_relationships(scene_struct: dict, eps: float = 0.2) -> dict:
    all_relationships = {}
    for name, direction_vec in scene_struct["directions"].items():
        if name in {"above", "below"}:
            continue
        relationships: List[List[int]] = []
        for i, obj1 in enumerate(scene_struct["objects"]):
            coords1 = obj1["3d_coords"]
            related = set()
            for j, obj2 in enumerate(scene_struct["objects"]):
                if obj1 is obj2:
                    continue
                coords2 = obj2["3d_coords"]
                diff = [coords2[k] - coords1[k] for k in range(3)]
                dot = sum(diff[k] * direction_vec[k] for k in range(3))
                if dot > eps:
                    related.add(j)
            relationships.append(sorted(related))
        all_relationships[name] = relationships
    return all_relationships


if __name__ == "__main__":
    if INSIDE_BLENDER:
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif {"-h", "--help"}.intersection(sys.argv):
        parser.print_help()
    else:
        print("Run this script from Blender, e.g.:")
        print("  blender --background --python render_images_modern.py -- [args]")
