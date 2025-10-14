# CLEVR Dataset Generator – To-Do List

These tasks capture follow-up improvements that go beyond the current cleanup.
Prioritise them based on your project needs.

## High priority
- **Automated tests inside Blender.** Add regression tests that exercise
  `render_scene` with a lightweight mock Cycles configuration to catch breaking
  API changes early.
- **Package distribution.** Wrap the image generation modules into an installable
  Python package so they can be reused without modifying `sys.path` manually.
- **Continuous integration.** Configure a CI workflow that lint-checks Python
  files and validates that documentation examples stay in sync.

## Medium priority
- **Scene asset validation.** Add a script that checks the integrity of
  `data/` assets (e.g. missing shapes or materials) before long rendering jobs.
- **Question template pruning.** Audit `question_generation/CLEVR_1.0_templates`
  and remove obsolete templates that are superseded by newer datasets.
- **Documentation automation.** Generate camera viewpoint diagrams from the JSON
  examples to make the README more visual.

## Low priority
- **GPU benchmarking.** Record render throughput for different GPU backends
  (CUDA, OPTIX, HIP, METAL) and resolutions to guide future optimisation.
- **Interactive demo.** Build a small notebook or web UI that lets users tweak
  camera/viewpoint JSON snippets and preview the resulting pose.
