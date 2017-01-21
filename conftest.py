# List of files that pytest should ignore

collect_ignore = ["setup.py",
                  "skimage/io/_plugins",
                  "doc/"]
try:
    import visvis
except ImportError:
    collect_ignore.append("skimage/measure/mc_meta/visual_test.py")

