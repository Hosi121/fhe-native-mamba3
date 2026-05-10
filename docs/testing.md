# Testing Strategy

This project uses a TDD-ish split because not every low-level path can run in
ordinary pre-commit checks.

## Fast Tests

Run:

```bash
scripts/run_checks.sh
```

This executes:

- `ruff format --check`
- `ruff check`
- `pytest` with coverage when `pytest-cov` is installed
- `pre-commit run --all-files`

Coverage is measured for the Python library code, excluding `cli.py` because
the CLI tests intentionally exercise it through subprocesses. The current
minimum is 70%.

## Native C++ Unit Tests

The native Stage 0 FIDESlib kernel has a small FIDESlib-free layout test:

```bash
cmake -S native/fideslib_stage0 -B build/stage0-layout-tests \
  -DFHE_STAGE0_BUILD_KERNEL=OFF \
  -DFHE_STAGE0_BUILD_TESTS=ON
cmake --build build/stage0-layout-tests
ctest --test-dir build/stage0-layout-tests --output-on-failure
```

`pytest` runs this automatically through `tests/test_native_layout_cpp.py`.
These tests cover the pure pieces that are easiest to break:

- rank-major slot layout
- rank-reduce and rank-local rotation-key inventory
- reduce/scatter masks
- dense and rank-local output slot mapping
- JSON emission for nonfinite decrypted values

## GPU Integration Probes

B200/FIDESlib runs are not part of ordinary pre-commit because they require a
SLURM allocation:

```bash
ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch slurm/fideslib_stage0.sbatch'
ssh high 'cd ~/cipher/fhe-native-mamba3 && sbatch slurm/fideslib_stage0_sweep.sbatch'
```

These probes produce benchmark JSON and are recorded in
`docs/probes/2026-05-10-b200-fideslib.md`.

## Current Gaps

- The native encrypted kernel itself is still verified by SLURM probes, not by
  a local C++ test runner.
- `rank-reduce` readout is verified up to `mimo_rank=2` under the toy CKKS
  parameters; higher ranks are recorded as known failing configurations.
  `rank-local` is the scatter-free candidate path for the next B200 sweep.
- Bootstrap scheduling has symbolic tests and a JSON-emitting OpenFHE bootstrap
  latency probe. Full end-to-end recurrence execution with inserted bootstraps
  is still the next integration gap.
