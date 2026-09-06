# Frozen photo rejection diagnostics

Prepared before final-model inference. The 14 cases comprise three unsupported
patterns with visible A–Z row context, three captures containing two natural rows,
three guides cutting the terminal cell, three groups containing 14–15 consecutive
cells, and two generated blank images. No model was loaded while selecting,
rendering, or auditing these fixtures.

The canonical images and manifest are in
`.braille-training-work/word-scan-v1-20260905/rejection-fixtures-v3/`.
`rejection-fixtures.json` is a byte-identical tracked copy of that manifest.
Its SHA-256 is
`4d92222c0fe73691efaeeba278252741b739cd04491121abb5ab579e25170f89`.
Earlier `v1`/`v2` directories are incomplete preparation attempts and must not be
used for inference. Their failures were a contact-sheet coordinate expression and
an insufficient number of eligible 13-cell prefixes, respectively; neither ran a
model. The final long cases use complete natural groups, retaining every cell.

All 12 photographic cases come from six photos in the existing held validation
families, `books/ola` and `handwritten/ang_redmi`. Source JPEG and CSV hashes,
one-based annotation rows, source boxes, masks, intersecting context annotations,
crop completeness, and normalized guide coordinates are recorded per case. Crops
preserve the source pixels and use the same aspect-2.5 context rule, an outward
integer crop, and Pillow RGB bilinear resize to 480 × 192. There is no recoloring,
denoising, composition, rotation, or inferred label repair.

`rejection-fixture-audit.json` verifies all 14 saved pixel arrays exactly, every
photographic source/split/hash/CSV row/mask, crop/guide math, all intersecting
annotations, and the intended rejection conditions. Box coordinate error is zero.
`rejection-fixture-review.json` records visual inspection and the unchanged hashes
of the existing 135, 405, and 607 fixture manifests, plus the separate 66-block and
44-negative regression inventories. These cases do not replace or enlarge those
benchmarks.

These are correlated diagnostic cases, not an independent holdout or physical
camera evidence. The three two-row photos contain closely related printed
content. The over-limit groups also contain unsupported masks, so a specific
length-only rejection reason is not required. Unsupported masks 60 and 47 are
shown alongside other complete cells; no isolated single-dot case is treated as
unambiguous dot-position ground truth. Visual review was performed by Codex, not
by an independent human annotator.

Acceptance for this diagnostic set is **zero accepted words out of 14**. Any
nonempty result fails, including a plausible shorter word. A rejection may come
from image quality, uncertainty, unsupported pattern, geometry, or length. Report
counts by category separately from whole-word benchmark accuracy. Do not tune
thresholds or checkpoints using these diagnostics and do not drop failed cases.
Recognition results remain pending until the final frozen model runs them.

Recheck the fixture audit from the repository root:

```powershell
C:/bm-envs/word-scan-torch-only/Scripts/python.exe training/scripts/word_rejection_audit.py --manifest .braille-training-work/word-scan-v1-20260905/rejection-fixtures-v3/manifest.json --source-split .braille-training-work/word-scan-v1-20260905/data/source-split.json --report .braille-training-work/word-scan-v1-20260905/rejection-fixtures-v3/audit-rerun.json
```

The final-model runner is the existing `training/scripts/word_diagnostics.py`:

```powershell
C:/bm-envs/angelina-benchmark/Scripts/python.exe training/scripts/word_diagnostics.py --run <frozen-candidate-directory> --fixtures .braille-training-work/word-scan-v1-20260905/rejection-fixtures-v3/manifest.json --report-name rejection-diagnostics.json
```

Use the candidate that has passed its Keras–TFLite conversion check. The runner
validates every input image hash, the exported model hash and parity flag, applies
the original RGB image-quality check, then uses the frozen thresholds. It refuses
to overwrite an existing diagnostic report.

The generator is `training/scripts/word_rejection_fixtures.py`. Its exact helper
sources are retained under `rejection-fixtures-v3/generator-source/` because the
training agent continues modifying `word_data.py`. The hash-identical original
helper was recovered from the retained run10 source archive; no current training
file was reverted. To reproduce the fixture manifest exactly, run that archived
generator with the original source split and a new empty output directory.
An actual regeneration reproduced the manifest and all 38 PNG files byte for
byte (14 model inputs, 12 source crops, and 12 annotated reviews).
