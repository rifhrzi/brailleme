# Adjacent grid centers can shorten a word

This source-level diagnostic reproduces a decoder defect, without model inference. Two fully visible, disjoint 4 by 12 pixel cells at adjacent stride-4 sites encode AB, but the old 3 by 3 local-maximum filter accepts A. Equal and unequal high scores both reproduce it. Moving the second cell two grid sites away restores AB. Ordinary coordinate and A-Z bit roundtrips pass.

The separate ideal405 audit finds zero adjacent or same-site distinct annotated pairs in the frozen validation set. It reads saved quality flags and cache identity metadata, not model predictions or image pixels. Therefore this latent small-cell defect is not a demonstrated cause of the current validation errors or physical C/H/J results.

These reports precede any fix. Geometry and dot outputs outside supervised centers may be unreliable; a replacement postprocessor requires tests and actual cached-output comparison. No model, thresholds, benchmark, application source, or active assets were changed by this audit.
