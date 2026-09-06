# Run27 continuation preflight evidence

The [preflight record](preflight.json) retains the run27 continuation protocol, independent command/reset guard, parent checkpoint HDF5 audit and exact 22 training sources. Its `ancestors` entry references the immutable [original run26 preflight](../horizontal-aspect-preflight/preflight.json), with separately retained run26 sampling history and lineage. Historical audits are not relabeled as tests of the new parent.

The parent is run26 epoch 10 (`6113867b21316669169391fa7216f9d1b9bf147a13c153889ff6de1c26e9a4ea`), which was eligible but unselected against the v15 control. The protocol intentionally resets Adam, epoch/order cursors and random streams; it repeats the same augmentation realizations. Local epochs 8–10 must satisfy their own sampling cycle.

This archive was assembled after launch from the original prelaunch records. No tests, model inference, training, conversion or physical-camera checks were rerun to assemble it. The retained verifier unit-test report is a separate source-integrity proof; real retained-data replay is appended separately. No active model or deliverable APK is established.

`archive-index.json` hashes every file in this new directory; links to the original run26 tree are validated by its unchanged archive index. No checkpoint binary is included.
