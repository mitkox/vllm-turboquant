# TurboQuant on RTX A6000 and CUDA 12.8

This document covers the supported private-workstation bring-up flow for
TurboQuant on `vLLM v0.19.0` with `RTX A6000 / SM86`.

## Supported CUDA GPUs

TurboQuant currently supports:

- `RTX A6000 / SM86`
- `GB10 / SM121`

The TurboQuant CLI surface is unchanged:

- `--kv-cache-dtype turboquant25`
- `--kv-cache-dtype turboquant35`
- `--enable-turboquant`
- `--turboquant-metadata-path /path/to/turboquant_kv.json`

## Environment

Use a source build and do not rely on precompiled vLLM wheels.

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install -r requirements/lint.txt
pre-commit install

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_PRECOMPILED=0
export VLLM_MAIN_CUDA_VERSION=12.8

uv pip install -e .
uv pip install -r requirements/test.txt
```

## Metadata generation

Generate TurboQuant metadata against an unquantized calibration model when the
target model is already quantized.

```bash
.venv/bin/python benchmarks/generate_turboquant_metadata.py \
  --target-model /models/target \
  --calibration-model /models/base \
  --recipe turboquant35 \
  --output /models/target/turboquant_kv.json
```

## TP=4 serving profile

For the 4x A6000 workstation, start with one server across all four GPUs.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

.venv/bin/vllm serve /models/target \
  --tensor-parallel-size 4 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype turboquant35 \
  --enable-turboquant \
  --turboquant-metadata-path /models/target/turboquant_kv.json
```

## Validation

Run the TurboQuant GPU subset for both recipes after the source build:

```bash
pytest tests/quantization/test_turboquant.py -v -s -k "turboquant25 or turboquant35"
```
