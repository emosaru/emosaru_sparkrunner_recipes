#!/bin/bash
set -e

# On DGX Spark (Grace Blackwell unified memory), pynvml.nvmlDeviceGetMemoryInfo
# returns NVMLError_NotSupported. Gemma4's _encoder_max_batch calls
# current_platform.get_device_total_memory() during profile_run(), which hits
# this path and crashes. Patch gemma4_mm.py to fall back to torch when NVML fails.

GEMMA4_MM=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_mm.py

python3 << 'EOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_mm.py"
with open(path, 'r') as f:
    src = f.read()

old = '            total_mem = current_platform.get_device_total_memory()'
new = (
    '            try:\n'
    '                total_mem = current_platform.get_device_total_memory()\n'
    '            except Exception:\n'
    '                import torch\n'
    '                total_mem = torch.cuda.get_device_properties(0).total_memory'
)

if old not in src:
    print("Pattern not found in gemma4_mm.py — already patched or version mismatch, skipping")
    sys.exit(0)

src = src.replace(old, new, 1)
with open(path, 'w') as f:
    f.write(src)
print("Patched gemma4_mm.py: _encoder_max_batch NVML fallback for unified-memory platforms")
EOF
