from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
        import triton
    except ImportError as exc:
        print(f"missing runtime dependency: {exc}", file=sys.stderr)
        return 1

    print(f"torch: {torch.__version__}")
    print(f"triton: {triton.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    print(f"cuda_device_name: {torch.cuda.get_device_name(device)}")
    print(f"cuda_capability: {torch.cuda.get_device_capability(device)}")

    x = torch.ones((16,), device=device)
    y = x * 2
    torch.cuda.synchronize()
    if float(y.sum().item()) != 32.0:
        print("CUDA tensor sanity check failed.", file=sys.stderr)
        return 1

    print("cuda_tensor_check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
