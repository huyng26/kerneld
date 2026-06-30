from __future__ import annotations


def rmsnorm_ref(x, weight, eps: float):
    input_dtype = x.dtype
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    y = x_float * (variance + eps).rsqrt()
    return weight * y.to(dtype=input_dtype)
