# NOTE:
# Temporary solution while we figure out how to add CUDA backend to the CI

import cupy as cp
import inspect
import skimage.morphology as morph


morph_functions = [
    func
    for submodule in ("binary", "gray")
    for name, func in inspect.getmembers(getattr(morph, submodule), inspect.isfunction)
    if not name.startswith("_") and "footprint" not in name
]

cu_arr = cp.zeros((10, 10))

print(f"Found {len(morph_functions)}.")

for morph_func in morph_functions:
    try:
        res = morph_func(cu_arr)
        assert isinstance(res, cp.ndarray)
    except Exception as e:
        print(f"Failed to call {morph_func}: {e}")
        continue
