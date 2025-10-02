# CLEVR Dataset Generation

This is the code used to generate the [CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/) as described in the paper:

**[CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](http://cs.stanford.edu/people/jcjohns/clevr/)**
 <br>
 <a href='http://cs.stanford.edu/people/jcjohns/'>Justin Johnson</a>,
 <a href='http://home.bharathh.info/'>Bharath Hariharan</a>,
 <a href='https://lvdmaaten.github.io/'>Laurens van der Maaten</a>,
 <a href='http://vision.stanford.edu/feifeili/'>Fei-Fei Li</a>,
 <a href='http://larryzitnick.org/'>Larry Zitnick</a>,
 <a href='http://www.rossgirshick.info/'>Ross Girshick</a>
 <br>
 Presented at [CVPR 2017](http://cvpr2017.thecvf.com/)

Code and pretrained models for the baselines used in the paper [can be found here](https://github.com/facebookresearch/clevr-iep).

You can use this code to render synthetic images and compositional questions for those images, like this:

<div align="center">
  <img src="images/example1080.png" width="800px">
</div>

**Q:** How many small spheres are there? <br>
**A:** 2

**Q:**  What number of cubes are small things or red metal objects? <br>
**A:**  2

**Q:** Does the metal sphere have the same color as the metal cylinder? <br>
**A:** Yes

**Q:** Are there more small cylinders than metal things? <br>
**A:** No

**Q:**  There is a cylinder that is on the right side of the large yellow object behind the blue ball; is there a shiny cube in front of it? <br>
**A:**  Yes

If you find this code useful in your research then please cite

```
@inproceedings{johnson2017clevr,
  title={CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning},
  author={Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens
          and Fei-Fei, Li and Zitnick, C Lawrence and Girshick, Ross},
  booktitle={CVPR},
  year={2017}
}
```

All code was developed and tested on OSX and Ubuntu 16.04.

## Step 1: Generating Images
First we render synthetic images using [Blender](https://www.blender.org/), outputting both rendered images as well as a JSON file containing ground-truth scene information for each image.

Blender ships with its own installation of Python which is used to execute scripts that interact with Blender; you'll need to add the `image_generation` directory to Python path of Blender's bundled Python. The easiest way to do this is by adding a `.pth` file to the `site-packages` directory of Blender's Python, like this:

```bash
echo $PWD/image_generation >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth
```

where `$BLENDER` is the directory where Blender is installed and `$VERSION` is your Blender version; for example on OSX you might run:

```bash
echo $PWD/image_generation >> /Applications/blender/blender.app/Contents/Resources/2.78/python/lib/python3.5/site-packages/clevr.pth
```

You can then render some images like this:

```bash
cd image_generation
blender --background --python render_images.py -- --num_images 10
```

### Modern viewpoint control

For workflows targeting Blender 3.x we provide an updated generator with
explicit camera control.  It accepts either a JSON list of camera locations or
samples viewpoints randomly from spherical coordinates.  This makes it easy to
render consistent multi-view datasets or explore diverse perspectives:

```bash
cd image_generation
blender --background --python render_images_viewpoints.py -- \
  --num_images 10 \
  --camera_viewpoints_json my_viewpoints.json
```

When no JSON file is supplied, the script will sample viewpoints uniformly
within the ranges defined by `--camera_radius_range`, `--camera_azimuth_range`
and `--camera_elevation_range`.  Each entry in `my_viewpoints.json` should be a
dictionary with at least a `"location"` field and optional `"look_at"` and
`"up"` keys, for example:

```json
[
  {"location": [7.5, 0.0, 6.0], "look_at": [0, 0, 0]},
  {"location": [-6.0, -6.0, 5.5], "look_at": [0, 0, 0]}
]
```

On OSX the `blender` binary is located inside the blender.app directory; for convenience you may want to
add the following alias to your `~/.bash_profile` file:

```bash
alias blender='/Applications/blender/blender.app/Contents/MacOS/blender'
```

If you have a supported GPU you can accelerate rendering dramatically.  For the
classic script use the legacy flag, while the modern script exposes explicit
backend selection:

```bash
# Original generator (CUDA only)
blender --background --python render_images.py -- --num_images 10 --use_gpu 1

# Modern generator with backend selection (CUDA, OPTIX, HIP, METAL)
blender --background --python render_images_viewpoints.py -- \
  --num_images 10 --use_gpu --gpu_backend CUDA
```

When running on a headless machine make sure the Blender binary can see your
GPU drivers.  The script automatically enables the Cycles backend and activates
all available devices that match the selected backend.  To maximize throughput
consider lowering `--render_num_samples` and tuning `--render_tile_size` to the
sweet spot of your GPU (smaller tiles for older cards, larger tiles for modern
architectures).

After this command terminates you should have ten freshly rendered images stored in `output/images` like these:

<div align="center">
  <img src="images/img1.png" width="260px">
  <img src="images/img2.png" width="260px">
  <img src="images/img3.png" width="260px">
  <br>
  <img src="images/img4.png" width="260px">
  <img src="images/img5.png" width="260px">
  <img src="images/img6.png" width="260px">
</div>

The file `output/CLEVR_scenes.json` will contain ground-truth scene information for all newly rendered images.

You can find [more details about image rendering here](image_generation/README.md).

## Step 2: Generating Questions
Next we generate questions, functional programs, and answers for the rendered images generated in the previous step.
This step takes as input the single JSON file containing all ground-truth scene information, and outputs a JSON file 
containing questions, answers, and functional programs for the questions in a single JSON file.

You can generate questions like this:

```bash
cd question_generation
python generate_questions.py
```

The file `output/CLEVR_questions.json` will then contain questions for the generated images.

You can [find more details about question generation here](question_generation/README.md).
