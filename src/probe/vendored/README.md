# Vendored third-party code

This directory contains third-party code vendored into this repository.

## sam/

Vendored from https://github.com/mazurowski-lab/mri_foundation at commit
`0f4f452614013873f8bb58a1fd613dbe36b58aba` (master branch as of 2026-04-21).

Original upstream path: `models/sam/`. The `models/` prefix is dropped; the
subtree is placed here under the `vendored/sam/` namespace. All imports within
the vendored code are relative (`from .modeling import ...`) and therefore
work unchanged under `src.probe.vendored.sam.*`.

Upstream license: Apache-2.0. The upstream repository advertises Apache-2.0 in
its README but does not ship a `LICENSE` file at the repository root. A copy of
the standard Apache-2.0 license text is included in this directory as `LICENSE`
to satisfy the redistribution requirement.

The MRI-CORE `models/sam/` subtree is itself a fork/derivative of Meta's
Segment Anything (https://github.com/facebookresearch/segment-anything), which
is also Apache-2.0. Per-file copyright headers from Meta Platforms are
preserved verbatim in the vendored sources.

### Local modifications

- `sam/build_sam.py`: fixed an upstream typo in `build_sam_vit_h`'s signature
  — the parameter was spelled `pretrain_sam` while the body referenced
  `pretrained_sam`, which would raise `NameError` if the function were ever
  called. Aligned the signature with `build_sam_vit_l` / `build_sam_vit_b`.
  Does not affect the `vit_b` code path we use.

Other files are copied byte-for-byte from upstream.

### Required dependencies

Importing `sam.sam_model_registry` triggers the following import chain:
`sam/__init__.py` → `.build_sam` → `.modeling` (__init__ imports all five
components eagerly) → `modeling/mask_decoder.py` → `modeling/vit.py`. Because
`vit.py` runs its module-level imports at class-definition time, the following
packages must be installed even when we only use the image encoder:
`torch`, `torchvision`, `einops`, `pillow`, `numpy`. `timm` is only needed by
`modeling/tiny_vit_sam.py`, which is NOT in the import chain for `vit_b`.
