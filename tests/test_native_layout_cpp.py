from __future__ import annotations

import subprocess


def test_native_stage0_layout_cpp_unit_tests(tmp_path) -> None:
    build_dir = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            "native/fideslib_stage0",
            "-B",
            str(build_dir),
            "-DFHE_STAGE0_BUILD_KERNEL=OFF",
            "-DFHE_STAGE0_BUILD_TESTS=ON",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
        check=True,
        capture_output=True,
        text=True,
    )
