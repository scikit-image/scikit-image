# List of files that pytest should ignore
collect_ignore = ["io/_plugins",]
try:
    import visvis
except ImportError:
    collect_ignore.append("measure/mc_meta/visual_test.py")
