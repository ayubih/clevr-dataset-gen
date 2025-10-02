#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 [--site-packages PATH]

Create a clevr.pth file inside Blender's bundled Python so that
image_generation modules are importable when Blender runs scripts.

Options:
  --site-packages PATH  Explicit path to Blender's site-packages directory.
                        Use this when Blender is not on PATH or automatic
                        detection fails.
USAGE
}

resolve_site_packages() {
  local supplied_path=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --site-packages)
        shift
        supplied_path="${1:-}"
        if [[ -z "$supplied_path" ]]; then
          echo "Error: --site-packages flag requires a value" >&2
          exit 1
        fi
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        echo "Unrecognised argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -n "$supplied_path" ]]; then
    echo "$supplied_path"
    return 0
  fi

  if command -v blender >/dev/null 2>&1; then
    local detected
    detected=$(blender -b --python-expr "import json, site; print(json.dumps(site.getsitepackages()))" 2>/dev/null | tail -n 1 || true)
    if [[ -n "$detected" ]]; then
      python - "$detected" <<'PY'
import json
import pathlib
import sys

raw = sys.argv[1]
paths = [pathlib.Path(p) for p in json.loads(raw)]
# Prefer directories that look like Blender bundles (contain 'blender' in path)
for candidate in paths:
    if 'blender' in str(candidate).lower():
        print(candidate)
        break
else:
    if paths:
        print(paths[0])
PY
      return 0
    fi
  fi

  echo "Error: Could not detect Blender's site-packages path." >&2
  echo "       Re-run with --site-packages /absolute/path/to/site-packages" >&2
  exit 1
}

SITE_PACKAGES=$(resolve_site_packages "$@")
if [[ ! -d "$SITE_PACKAGES" ]]; then
  echo "Error: $SITE_PACKAGES does not exist or is not a directory." >&2
  exit 1
fi

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PTH_FILE="$SITE_PACKAGES/clevr.pth"

printf '%s\n' "$REPO_ROOT/image_generation" | tee "$PTH_FILE" > /dev/null

echo "Wrote $PTH_FILE"
echo "Blender will now be able to import clevr image generation modules."
