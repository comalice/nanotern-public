#!/usr/bin/env bash
# gpu_smoke.sh — first GPU window: piece 1 benchmark + piece 5 smoke checks.
# 1. bf16 matmul TFLOPs benchmark (piece 1 remaining check)
# 2. 100-step train of the 126M smoke model (loss decreases, val bpb finite)
# 3. resume test: +50 more steps from the checkpoint (step/loss restore)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "=== piece 1: bf16 matmul benchmark ==="
$PY - <<'EOF'
import torch, time
assert torch.cuda.is_available()
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
b = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
for _ in range(3): c = a @ b
torch.cuda.synchronize()
t0 = time.time(); n = 20
for _ in range(n): c = a @ b
torch.cuda.synchronize()
dt = time.time() - t0
tflops = 2 * 8192**3 * n / dt / 1e12
print(f"bf16 matmul 8192^3: {tflops:.1f} TFLOPs")
assert tflops > 50, "below 50 TFLOPs bar"
print("piece 1 check PASSED")
EOF

echo "=== piece 5a: 100-step smoke run ==="
$PY -m nanotern.train --config smoke --run-name smoke-bringup \
    --num-iterations 100 --eval-every 50 || exit 1

echo "=== piece 5b: resume +50 steps ==="
$PY -m nanotern.train --config smoke --run-name smoke-bringup \
    --num-iterations 150 --eval-every 50 --resume auto || exit 1

echo "=== gpu_smoke done ==="
grep -E "step|bpb|piece" logs/gpu_smoke.log 2>/dev/null | tail -20 || true
