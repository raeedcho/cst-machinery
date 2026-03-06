# Submodules are imported directly by each script that needs them.
# No eager imports here — lfads_callbacks, crystal_models, and ofc_model
# all pull in torch/pytorch_lightning/jax and should only load when needed.