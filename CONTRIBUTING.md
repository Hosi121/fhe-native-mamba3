# Contributing

This repository is an implementation-first research prototype. The working
standard is: every claim should eventually reduce to runnable code, measured
artifacts, and a named next bottleneck.

## Local Workflow

Install the development dependencies and pre-commit hook:

```bash
python3 -m pip install --user -e '.[dev]'
pre-commit install
```

Before committing, run:

```bash
scripts/run_checks.sh
```

This is the required local gate for ordinary changes. GPU/SLURM probes are
tracked separately because they require the `high` cluster and a paid B200
allocation.

## Versioning

Use SemVer. Do not use `version1`, `version2`, date-only names, or milestone
names as package versions.

- Patch bumps are for bug fixes, test additions, process/docs changes, and
  narrow API hardening inside the current stage.
- Minor bumps are for a new runnable capability boundary, for example a real
  checkpoint-to-bundle-to-encrypted-smoke path.
- `1.0.0` is reserved for loading existing OSS weights and running an
  end-to-end encrypted inference path with benchmark output.

Current expected sequence:

- `0.2.x`: backend abstraction, Stage 0 harnesses, layout safety, status gates.
- `0.3.x`: real Mamba checkpoint to bundle to encrypted recurrence smoke.
- `0.4.x`: multi-layer/24-layer recurrence smoke with bootstrap scheduling.
- `0.5.x`: reproducible OpenFHE/FIDESlib Stage 0 benchmark.
- `1.0.0`: OSS baseline usable by external users.

## PBI Standard

A PBI should include:

- stage: Stage 0, Stage 1, or Stage 2,
- priority: P0, P1, P2,
- dependencies and blockers,
- parallelization notes,
- acceptance checks,
- benchmark or artifact output, when relevant,
- next bottleneck expected after completion.

Definition of Ready:

- the write scope is clear,
- dependencies are explicit,
- there is a runnable acceptance command or artifact target,
- the expected failure mode is named.

Definition of Done:

- code and tests are committed,
- `scripts/run_checks.sh` passes,
- benchmark/probe JSON is attached or its absence is explicitly explained,
- README/docs/status are updated when behavior or claims change,
- the next bottleneck is added to the issue or linked PBI.

## Benchmark Artifacts

Benchmark JSON should include:

- repo version and commit,
- backend, hardware, and input mode,
- checkpoint or synthetic problem identifier,
- latency and operation counts,
- accuracy/error metric,
- bootstrap count and rotation key count when applicable,
- measurement scope and non-claims.

Use `runs/` for local artifacts and `docs/probes/` for curated probe notes. Do
not treat toy CKKS parameters as Stage 0 target measurements unless the artifact
explicitly says so.

## Review Checklist

Review low-level FHE code with these questions first:

- Does the ciphertext slot layout have a single explicit contract?
- Are required rotations reported by the same API that executes the layout?
- Can a ciphertext from one layout be accidentally passed as another layout?
- Are encrypted, plaintext, and tracking backend paths semantically aligned?
- Does a partial check clearly say it is partial?
- Does the public package API point at the current implementation?

For checkpoint work, avoid claiming full Mamba correctness unless the result is
compared against the actual checkpoint path being claimed.

