from __future__ import annotations


def rmsnorm_ref(x, weight, eps: float):
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    y = x_float * (variance + eps).rsqrt()
    return (y * weight).to(dtype=x.dtype)
