#!/usr/bin/env bash
# UUID / topology / software capture for one machine. Usage: ./capture_env.sh <card-label>
set -u
CARD=${1:?card label, e.g. rtx4090}
mkdir -p env
OUT=env/uuid_${CARD}.txt
{
  echo "# captured $(date -u '+%Y-%m-%dT%H:%M:%SZ') on $(hostname)"
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap,pcie.link.gen.current,pcie.link.width.current --format=csv
  nvidia-smi
  python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__)"
  python -c "import vllm;print('vllm',vllm.__version__)" 2>/dev/null || echo "vllm not installed"
  python -c "import bitsandbytes as b;print('bitsandbytes',b.__version__)" 2>/dev/null || echo "bitsandbytes not installed"
} > "$OUT" 2>&1
nvidia-smi topo -m > env/topo_${CARD}.txt 2>&1
echo "wrote $OUT and env/topo_${CARD}.txt"
