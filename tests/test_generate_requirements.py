import json

import pytest

from tools.generate_requirements import _min_version, update_asv_conf


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("numpy>=2.1", "2.1"),
        ("scipy>=1.15; sys_platform != \"emscripten\"", "1.15"),
        ("scipy>=1.14; sys_platform == \"emscripten\"", None),
        ("scipy>=1.15,<2", "1.15"),
        ("numpy", None),
        ("numpy==2.1", None),
    ],
)
def test_min_version(spec, expected):
    assert _min_version(spec) == expected


@pytest.mark.parametrize(
    "deps",
    [
        [
            "numpy>=2.1",
            "scipy>=1.15; sys_platform != \"emscripten\"",
            "scipy>=1.14; sys_platform == \"emscripten\"",
        ],
        [
            "scipy>=1.14; sys_platform == \"emscripten\"",
            "scipy>=1.15; sys_platform != \"emscripten\"",
            "numpy>=2.1",
        ],
    ],
)
def test_update_asv_conf(tmp_path, monkeypatch, deps):
    conf_path = tmp_path / "asv.conf.json"
    conf_path.write_text(json.dumps({"matrix": {"numpy": ["1.0"], "scipy": ["1.0"]}}))
    monkeypatch.setattr("tools.generate_requirements.repo_dir", tmp_path)

    update_asv_conf({"project": {"dependencies": deps}})

    matrix = json.loads(conf_path.read_text())["matrix"]
    assert matrix["numpy"] == ["2.1"]
    assert matrix["scipy"] == ["1.15"]
