# CLEVR Dataset Generator

This repository provides the cleaned-up tooling required to generate the
synthetic CLEVR dataset: Blender scripts that render scenes plus Python utilities
that compose question-answer pairs from the rendered metadata.

## Repository layout
- `image_generation/` – Blender-facing code, scene assets, and camera presets.
- `question_generation/` – Scripts and templates that expand the rendered scene
  graphs into compositional questions.
- `scripts/` – Helper utilities such as environment setup helpers for Blender.
- `docs/` – Project to-dos and supplementary documentation.

The renderer always boots from `image_generation/data/base_scene.blend`, the same
base scene that shipped with the original CLEVR release. It defines the ground
plane, light rig, and camera rig that every render builds upon.

## Quick setup
1. Install [Blender](https://www.blender.org/) (3.x or 4.x recommended).
2. Make sure Python 3 is available on your system for the question-generation
   scripts.
3. Allow Blender to import the image-generation modules by running:
   ```bash
   ./scripts/setup_blender.sh
   ```
   If Blender is not on your `$PATH`, supply the path to its bundled
   `site-packages` directory:
   ```bash
   ./scripts/setup_blender.sh --site-packages /Applications/blender/Blender.app/Contents/Resources/4.0/python/lib/python3.10/site-packages
   ```
   The script writes a `clevr.pth` file that points Blender's Python to the
   `image_generation` package.

## Rendering pipeline
Run the modern entry point from Blender to produce images and scene JSON:
```bash
cd image_generation
blender --background --python render_images_modern.py -- \
  --num_images 12 \
  --output_image_dir ../output/images \
  --output_scene_dir ../output/scenes \
  --output_scene_file ../output/CLEVR_scenes.json
```

### Camera viewpoint control and rotations
`render_images_modern.py` accepts `--camera_viewpoints_json` describing explicit
poses. Each entry supports:

| Field      | Description |
|------------|-------------|
| `location` | XYZ coordinates of the camera in world units. Changing the vector rotates the camera around the scene (e.g. rotating the XY components sweeps around the vertical axis). |
| `look_at`  | World-space target the camera focuses on. Alter it to tilt the camera up/down/sideways without changing its position. |
| `up`       | Up-direction vector for the camera. Modify this to roll the camera about its viewing axis. Defaults to `(0, 0, 1)`.

The helper file `image_generation/camera_viewpoints_rotations.json` contains
pre-baked examples:
- **`yaw_0_deg` / `yaw_90_deg`** – rotate the camera around the vertical axis by
  adjusting the XY position while keeping the target fixed.
- **`pitch_down_30_deg`** – raises the camera and lowers the focus point to pitch
  downward.
- **`roll_around_view_axis`** – uses a custom `up` vector to roll the frame.

Render with these presets and sequentially cycle through the poses:
```bash
blender --background --python render_images_modern.py -- \
  --num_images 4 \
  --camera_viewpoints_json camera_viewpoints_rotations.json \
  --camera_viewpoint_strategy sequential \
  --render_num_samples 128 \
  --output_image_dir ../output/rotations/images \
  --output_scene_dir ../output/rotations/scenes \
  --output_scene_file ../output/rotations/CLEVR_scenes.json
```

To generate different angles, create your own JSON by rotating a base location
vector. For example, to yaw around the vertical axis by 45° with radius `r`:
```python
import math
r = 6.0
yaw_deg = 45
location = [r * math.cos(math.radians(yaw_deg)),
            r * math.sin(math.radians(yaw_deg)),
            4.5]
```
Store the resulting coordinates in a new viewpoint entry.

### GPU acceleration
GPU rendering is optional but supported. Pass `--use_gpu` together with the
backend you want to activate:
```bash
blender --background --python render_images_modern.py -- \
  --num_images 64 \
  --use_gpu --gpu_device_type CUDA \
  --enable_adaptive_sampling --render_tile_size 1024
```
`--gpu_device_type` can be `CUDA`, `OPTIX`, `HIP`, `METAL`, or `NONE`. When the
flag is omitted the script sticks to CPU-only rendering.

### Legacy workflow
`render_images.py` mirrors the original CLEVR release for Blender 2.7x users and
still supports CUDA via `--use_gpu 1`, but the modern script above is the
recommended path.

## Generating questions
After rendering, convert the combined scenes JSON into questions:
```bash
cd question_generation
python generate_questions.py \
  --input_scene_file ../output/CLEVR_scenes.json \
  --output_questions_file ../output/CLEVR_questions.json
```

The templates under `question_generation/CLEVR_1.0_templates` define the core
question set. Adjust `--templates_per_image` and `--instances_per_template` to
tune output volume.

## Frequently asked points
- **Does the renderer require CUDA?** No. CUDA/OptiX/HIP/Metal are optional
  accelerators; CPU rendering remains supported out of the box.
- **What controls the base environment?** All renders start from
  `data/base_scene.blend`, which provides the ground plane, lights, and default
  camera before any randomisation or supplied viewpoints are applied.

For open tasks and future improvements see [`docs/TODO.md`](docs/TODO.md).
