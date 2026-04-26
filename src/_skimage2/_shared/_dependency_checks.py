import sys
import platform

is_wasm = (sys.platform == "emscripten") or (platform.machine() in ["wasm32", "wasm64"])
