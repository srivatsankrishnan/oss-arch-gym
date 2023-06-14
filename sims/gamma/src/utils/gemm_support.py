import numpy as np


def translate_to_gemm(model_defs):
    gemm_model_defs = []
    for layer in model_defs:
        K, C, Y, X, R, S, T = layer
        gemm_M = X * Y
        gemm_K = R * S * C
        gemm_N = K
        C = gemm_K
        Y = gemm_M
        K = gemm_N
        X = R = S = 1
        gemm_model_defs.append([K, C, Y, X, R, S, T])
    return np.array(gemm_model_defs)

def check_tpu(accel_cstr=None, map_cstr=None):
    if accel_cstr == "tpu_accel" or map_cstr == "tpu_map":
        return True
    else:
        return False