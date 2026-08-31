#!/usr/bin/env bash
# Download the SONIC checkpoint variants from HF nvidia/GEAR-SONIC.
# The release/ (v1.0) weights are committed to git; the low_latency and
# sonic_v1_1 decoders are ~150 MB (over GitHub's file limit), so they are
# gitignored and fetched by this script. Skips files already on disk.
set -euo pipefail
cd "$(dirname "$0")"

HF=https://huggingface.co/nvidia/GEAR-SONIC/resolve/main

fetch() {           # fetch <local path> <remote path>
  if [ -f "$1" ]; then echo "have  $1"; return; fi
  echo "fetch $1"
  mkdir -p "$(dirname "$1")"
  curl -SfL --retry 3 -o "$1" "$HF/$2"
}

fetch release/model_encoder.onnx        model_encoder.onnx
fetch release/model_decoder.onnx        model_decoder.onnx
fetch release/observation_config.yaml   observation_config.yaml

for v in low_latency sonic_v1_1; do
  for f in model_encoder.onnx model_decoder.onnx observation_config.yaml; do
    fetch "$v/$f" "$v/$f"
  done
done

echo done
