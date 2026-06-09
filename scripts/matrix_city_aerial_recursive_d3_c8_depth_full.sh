#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_recursive_d3_c8_depth.yaml}"
SOURCE_PARTITION_TREE="${SOURCE_PARTITION_TREE:-output/mc_aerial_recursive_d3_c8/partitions/partition_tree.json}"
CUDA_IDS="${CUDA_IDS:-2,3,4,5,6,7,8,9}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

RUN_PARTITION="${RUN_PARTITION:-0}"
RUN_TRAIN_BLOCKS="${RUN_TRAIN_BLOCKS:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_RENDER_EVAL="${RUN_RENDER_EVAL:-1}"
RUN_METRICS="${RUN_METRICS:-1}"

FORCE_PARTITION="${FORCE_PARTITION:-0}"
SYNC_PARTITION_TREE="${SYNC_PARTITION_TREE:-1}"
SKIP_TRAINED_BLOCKS="${SKIP_TRAINED_BLOCKS:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
MERGE_ALLOW_MISSING="${MERGE_ALLOW_MISSING:-0}"

EXTRA_PARTITION_ARGS="${EXTRA_PARTITION_ARGS:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_MERGE_ARGS="${EXTRA_MERGE_ARGS:-}"
EXTRA_RENDER_ARGS="${EXTRA_RENDER_ARGS:-}"

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

eval "$(
    python - "$CONFIG" <<'PY'
import os
import shlex
import sys

from utils.config_utils import get_in, load_yaml_config, partition_tree_path

cfg = load_yaml_config(sys.argv[1])
source_path = get_in(cfg, "dataset.source_path", "")
depths = get_in(cfg, "dataset.depths", "")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

emit("CFG_PARTITION_TREE", partition_tree_path(cfg))
emit("CFG_COARSE_MODEL", get_in(cfg, "model.coarse_model", ""))
emit("CFG_SOURCE_PATH", source_path)
emit("CFG_DEPTHS", depths)
emit("CFG_DEPTH_DIR", os.path.join(source_path, depths) if depths else "")
emit("CFG_DEPTH_PARAMS", os.path.join(source_path, "sparse/0/depth_params.json"))
emit("CFG_MAX_DEPTH", get_in(cfg, "partition.max_depth", 3))
emit("CFG_MAX_BLOCKS", get_in(cfg, "partition.max_blocks", 8))
PY
)"

PARTITION_TREE="$CFG_PARTITION_TREE"

if [ -z "$CFG_DEPTHS" ]; then
    echo "Depth block regularization requires dataset.depths in $CONFIG" >&2
    exit 1
fi
if [ ! -d "$CFG_DEPTH_DIR" ]; then
    echo "Depth directory not found: $CFG_DEPTH_DIR" >&2
    echo "Generate it with scripts/prepare_depth_regularization.sh before training." >&2
    exit 1
fi
if [ ! -f "$CFG_DEPTH_PARAMS" ]; then
    echo "Depth params not found: $CFG_DEPTH_PARAMS" >&2
    echo "Generate it with scripts/prepare_depth_regularization.sh before training." >&2
    exit 1
fi
if [ ! -f "$CFG_COARSE_MODEL" ]; then
    echo "Coarse model not found: $CFG_COARSE_MODEL" >&2
    exit 1
fi

partition_args=(
    --config "$CONFIG"
    --override "partition.max_depth=$CFG_MAX_DEPTH"
    --override "partition.max_blocks=$CFG_MAX_BLOCKS"
)
if [ -n "$EXTRA_PARTITION_ARGS" ]; then
    # shellcheck disable=SC2206
    partition_args+=($EXTRA_PARTITION_ARGS)
fi

if [ "$RUN_PARTITION" = "1" ]; then
    if [ -f "$PARTITION_TREE" ] && [ "$FORCE_PARTITION" != "1" ]; then
        echo "Partition tree already exists: $PARTITION_TREE"
        echo "Skipping partition. Set FORCE_PARTITION=1 to regenerate it."
    else
        echo "Running recursive partition with max depth $CFG_MAX_DEPTH and max $CFG_MAX_BLOCKS blocks..."
        python partition.py "${partition_args[@]}"
    fi
elif [ "$SYNC_PARTITION_TREE" = "1" ]; then
    if [ -f "$PARTITION_TREE" ] && [ "$FORCE_PARTITION" != "1" ]; then
        echo "Depth partition tree already exists: $PARTITION_TREE"
        echo "Skipping sync. Set FORCE_PARTITION=1 to refresh it from SOURCE_PARTITION_TREE."
    else
        if [ ! -f "$SOURCE_PARTITION_TREE" ]; then
            echo "Source partition tree missing: $SOURCE_PARTITION_TREE" >&2
            echo "Set RUN_PARTITION=1 to generate a new d3/c8 partition from $CONFIG." >&2
            exit 1
        fi
        echo "Copying existing d3/c8 partition tree and patching coarse_model..."
        python - "$SOURCE_PARTITION_TREE" "$PARTITION_TREE" "$CFG_COARSE_MODEL" <<'PY'
import json
import os
import sys

source_tree, output_tree, coarse_model = sys.argv[1:4]
coarse_model = os.path.abspath(coarse_model)
with open(source_tree, "r") as f:
    tree = json.load(f)

tree["coarse_model"] = coarse_model
config = tree.setdefault("config", {})
if isinstance(config, dict):
    config["coarse_model"] = coarse_model
    model = config.get("model")
    if isinstance(model, dict):
        model["coarse_model"] = coarse_model
    partition_output = os.path.dirname(os.path.abspath(output_tree))
    config["partition_output"] = partition_output
    partition = config.get("partition")
    if isinstance(partition, dict):
        partition["output_path"] = partition_output

os.makedirs(os.path.dirname(output_tree), exist_ok=True)
with open(output_tree, "w") as f:
    json.dump(tree, f, indent=2)
    f.write("\n")

print(f"wrote {output_tree}")
PY
    fi
else
    echo "RUN_PARTITION=0 and SYNC_PARTITION_TREE=0, reusing partition: $PARTITION_TREE"
fi

python - "$PARTITION_TREE" "$CFG_MAX_BLOCKS" <<'PY'
import json
import os
import sys

partition_tree = sys.argv[1]
expected_blocks = int(sys.argv[2])
if not os.path.isfile(partition_tree):
    raise SystemExit(f"Partition tree missing: {partition_tree}")

with open(partition_tree, "r") as f:
    tree = json.load(f)

blocks = tree.get("blocks", [])
coarse_model = tree.get("coarse_model") or tree.get("config", {}).get("coarse_model", "")
if len(blocks) != expected_blocks:
    raise SystemExit(f"Expected {expected_blocks} blocks, found {len(blocks)} in {partition_tree}")
if not os.path.isfile(coarse_model):
    raise SystemExit(f"Partition coarse_model is missing: {coarse_model}")

print(f"Partition ready: {partition_tree}")
print(f"Blocks: {len(blocks)}")
print(f"Coarse model: {coarse_model}")
PY

echo "Running d3/c8 depth block-reg training / merge / render eval / metrics..."
echo "Config: $CONFIG"
echo "Source partition: $SOURCE_PARTITION_TREE"
echo "Depth partition: $PARTITION_TREE"
echo "CUDA_IDS: $CUDA_IDS"
echo "MAX_PARALLEL: $MAX_PARALLEL"

CONFIG="$CONFIG" \
CUDA_IDS="$CUDA_IDS" \
MAX_PARALLEL="$MAX_PARALLEL" \
TRAIN_BLOCKS="$RUN_TRAIN_BLOCKS" \
MERGE_BLOCKS="$RUN_MERGE" \
RENDER_TEST_SET="$RUN_RENDER_EVAL" \
RUN_METRICS="$RUN_METRICS" \
SKIP_TRAINED_BLOCKS="$SKIP_TRAINED_BLOCKS" \
STOP_ON_FAILURE="$STOP_ON_FAILURE" \
MERGE_ALLOW_MISSING="$MERGE_ALLOW_MISSING" \
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS" \
EXTRA_MERGE_ARGS="$EXTRA_MERGE_ARGS" \
EXTRA_RENDER_ARGS="$EXTRA_RENDER_ARGS" \
bash scripts/matrix_city_aerial_train_blocks.sh
