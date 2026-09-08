# SO-101 workstation source snapshot

This branch contains the Evomind application source deployed on the SO-101 workstation, captured on 2026-09-09. Its deployed base release was `519cf031-lr061-t211-cu128-b1`, with subsequent workstation fixes included in the source tree. `release-manifest.json` is the original base-release record, not a validation report for this new Git commit.

The snapshot includes SO-101 hardware recovery, episode recovery, OpenPI JAX policy integration, and the deployed web application source. It is a separate deployment line from `main`; files and features absent from this workstation were not copied from newer main revisions. The local `codex/pedal-and-replay` work is independent and is not included in this snapshot.

Model checkpoints, datasets, virtual environments, `node_modules`, compiled web output, local logs, credentials, and deployment backups are excluded. Original documentation links from the repository are retained. No Git LFS is used.

OpenPI numerical model servers and checkpoint-specific launch configurations live outside the Evomind application directory on the workstation. They must be supplied separately when deploying this branch; the included `openpi_jax` policy is the LeRobot-side adapter, not a bundled JAX environment or model checkpoint.

Before publication, the imported Python source was syntax-checked and the publishable files scanned for credentials. This import does not run robot motion or assert that a fresh host is hardware-validated.
