#!/bin/bash
# Generate the 128x128 synthetic stereo dataset (left-view GT) used by the configs.
#
#   bash scripts/generate_data.sh train [OUT_DIR]   # 25000 images -> data/stereo_128_train
#   bash scripts/generate_data.sh test  [OUT_DIR]   #   100 images -> data/stereo_128_test
#
# Train composition:  5000 one-object  +  10000 two-object (different colour)
#                                      +  10000 two-object (same colour); all different depth.
# Test composition:   100 two-object (different colour, different depth).
#
# Textures: objects use the 'banded' texture class and backgrounds the 'woven' class of the
# Describable Textures Dataset (DTD, https://www.robots.ox.ac.uk/~vgg/data/dtd/). Put them in
#   $TEX_DIR (default data_gen/texture_imgs/banded)  and  $BGTEX_DIR (default data_gen/texture_imgs/woven)
# If either directory is missing or empty, procedural textures (checker/stripes/dots) are used.
#
# Environment overrides: NUM_WORKERS (parallel generator processes per group), PYTHON.
set -euo pipefail

SPLIT="${1:?usage: bash scripts/generate_data.sh <train|test> [OUT_DIR]}"
OUT="${2:-data/stereo_128_${SPLIT}}"
PYTHON="${PYTHON:-python}"
TEX_DIR="${TEX_DIR:-data_gen/texture_imgs/banded}"
BGTEX_DIR="${BGTEX_DIR:-data_gen/texture_imgs/woven}"

FOCAL_PX=300.0      # halved from the 256px value so disparity/width ratio is unchanged
CLIP_DMAX=64        # disparity ceiling (px) for the UDF / field channel
SHAPES="triangle square pentagon hexagon heptagon octagon"

case "$SPLIT" in
  train) N_ONE_WORKERS=5;  ONE_CHUNK=1000; N_TWO_WORKERS=10; TWO_CHUNK=1000; SAME_OFFSET=10000 ;;
  test)  N_ONE_WORKERS=0;  ONE_CHUNK=0;    N_TWO_WORKERS=10; TWO_CHUNK=10;   SAME_OFFSET=-1 ;;
  *) echo "split must be 'train' or 'test'" >&2; exit 1 ;;
esac
N_ONE_WORKERS="${NUM_WORKERS:-$N_ONE_WORKERS}"; [[ "$SPLIT" == test ]] && N_ONE_WORKERS=0

TEX_ARGS=(--mode-texture mix --p-texture 0.5)
if compgen -G "$TEX_DIR/*" >/dev/null && compgen -G "$BGTEX_DIR/*" >/dev/null; then
  TEX_ARGS+=(--texture-dir "$TEX_DIR" --bg-texture all --bg-texture-dir "$BGTEX_DIR")
else
  echo "[generate_data] texture dirs empty/missing -> procedural textures only" >&2
fi

GEN=(
  "$PYTHON" data_gen/shape_stereo_slant.py
  --output-dir "$OUT" --stereo-mode slanted --boundary-mode area --area-supersample 8
  --image-size 128 128 --focal-px "$FOCAL_PX" --z-min 0.5 --z-max 3.0 --clip-dmax "$CLIP_DMAX"
  --p-zero-disp 0.0 --shapes $SHAPES "${TEX_ARGS[@]}" --no-viz
)
TWO=(--mode two --same-disp-thresh 1.0 --p-large-overlap 0.5 --min-overlap-ratio 0.3 --p-same-depth 0.0)

mkdir -p "$OUT" logs
pids=()

for ((w = 0; w < N_ONE_WORKERS; w++)); do          # one-object: indices 0..4999
  off=$((w * ONE_CHUNK))
  "${GEN[@]}" --mode one --n-one "$ONE_CHUNK" --idx-offset "$off" > "logs/gen_one_${off}.log" 2>&1 &
  pids+=($!)
done
for ((w = 0; w < N_TWO_WORKERS; w++)); do         # two-object, different colour
  off=$((w * TWO_CHUNK))
  "${GEN[@]}" "${TWO[@]}" --n-two-diff "$TWO_CHUNK" --n-two-same 0 --idx-offset "$off" \
      > "logs/gen_two_diff_${off}.log" 2>&1 &
  pids+=($!)
done
if (( SAME_OFFSET >= 0 )); then                    # two-object, same colour: indices 10000..19999
  for ((w = 0; w < N_TWO_WORKERS; w++)); do
    off=$((SAME_OFFSET + w * TWO_CHUNK))
    "${GEN[@]}" "${TWO[@]}" --n-two-diff 0 --n-two-same "$TWO_CHUNK" --idx-offset "$off" \
        > "logs/gen_two_same_${off}.log" 2>&1 &
    pids+=($!)
  done
fi

fails=0
for pid in "${pids[@]}"; do wait "$pid" || fails=$((fails + 1)); done
echo "[generate_data] done: $OUT  (failed workers: $fails; see logs/gen_*.log)"
exit $((fails > 0))
